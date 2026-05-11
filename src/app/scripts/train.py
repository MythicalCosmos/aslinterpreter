import os
import numpy as np
import mediapipe as mp
import cv2
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
import json
from datetime import datetime
import zipfile
import json
import shutil
import tempfile

SCRIPT_DIR   = Path(__file__).resolve().parent          # src/app/scripts/
SHARED_DIR   = SCRIPT_DIR.parent                        # src/app/
DATASET_PATH = SHARED_DIR / "datasets"                  # src/app/datasets/
EXPORT_PATH  = SHARED_DIR / "exports"                   # src/app/exports/
DEPLOY_DIR   = SCRIPT_DIR.parent.parent / "deploy"      # src/app/deploy/
MODEL_OUT    = DEPLOY_DIR / "asl_model.tflite"
LABELS_OUT   = DEPLOY_DIR / "labels.txt"
LOG_OUT      = SHARED_DIR.parent / "logs" / "training_log.txt"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

def log(msg):
    """Print and also write to the log file the GUI reads."""
    print(msg)
    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_OUT, "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")

# ---------------------------------------------------------------------------
# Stage 1 — Extract landmarks from every image using MediaPipe
# ---------------------------------------------------------------------------
def extract_landmarks(image_path, hands):
    """
    Run MediaPipe Hands on one image and return a flat 63-float vector.
    The 21 landmarks each have x, y, z — all normalized relative to the
    wrist so the gesture looks the same no matter where in the frame the
    hand appears.
    Returns None if no hand is detected in the image.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0].landmark
    wrist = lm[0]
    # Subtract wrist position so the vector describes hand SHAPE not position
    return np.array([
        [l.x - wrist.x, l.y - wrist.y, l.z - wrist.z]
        for l in lm
    ]).flatten()  # shape: (63,)

def collect_dataset():
    """
    Walk every subfolder of datasets/.
    Folder name = label (e.g. 'A', 'B', ...).
    Returns X (n_samples, 63), y (n_samples,), label_names (list of str).
    """
    label_names = sorted([
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(DATASET_PATH / d)
    ])
    if not label_names:
        raise RuntimeError(f"No gesture folders found in {DATASET_PATH}")

    log(f"Found {len(label_names)} gesture folders for training.")
    log(f"Labels: {label_names}")

    X, y = [], []

    # Use static_image_mode=True here — faster and more accurate for still images
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3   # low threshold so we don't skip too many images
    ) as hands:
        for label_idx, label in enumerate(label_names):
            folder = DATASET_PATH / label
            images = [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS)]
            good = 0
            for fname in images:
                vec = extract_landmarks(folder / fname, hands)
                if vec is not None:
                    X.append(vec)
                    y.append(label_idx)
                    good += 1
            log(f"  {label}: {good}/{len(images)} images had a detectable hand")

    if len(X) == 0:
        raise RuntimeError("No landmarks could be extracted. Check your images.")

    return np.array(X), np.array(y), label_names

# ---------------------------------------------------------------------------
# Stage 2 — Train a Random Forest classifier on the landmark vectors
# ---------------------------------------------------------------------------
def train_classifier(X, y, label_names):
    """
    Train a Random Forest on the 63-float landmark vectors.
    Random Forest is chosen because:
      - Works on every OS with no GPU needed
      - Trains in seconds even on large datasets
      - Very accurate for this kind of structured numeric data
      - No hyperparameter tuning required to get good results
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples")

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    log(f"Validation accuracy: {acc:.2%}")

    return clf, X_train, y_train

# ---------------------------------------------------------------------------
# Stage 3 — Export to TFLite so the existing inference worker loads it
# ---------------------------------------------------------------------------
def export_tflite(clf, X_train, y_train, label_names):
    """
    Builds a small Keras model that mimics the Random Forest,
    converts it to TFLite, and saves it to src/deploy/asl_model.tflite.
    Returns the raw tflite bytes so export_task can reuse them.
    """
    n_classes = len(label_names)
    proba = clf.predict_proba(X_train)

    inp = tf.keras.Input(shape=(63,), name="landmarks")
    x = tf.keras.layers.Dense(128, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax", name="gesture")(x)
    model = tf.keras.Model(inputs=inp, outputs=out)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    log("Training TFLite export model (this takes about 30 seconds)...")
    model.fit(X_train, proba, epochs=80, batch_size=32, verbose=0)
    log("Keras model trained")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.write_bytes(tflite_bytes)
    log(f"TFLite model saved → {MODEL_OUT}")

    LABELS_OUT.write_text("\n".join(label_names))
    log(f"Labels saved → {LABELS_OUT}")

    # Return the bytes so export_task can bundle them
    return tflite_bytes

def export_task(tflite_bytes, label_names):
    """
    Produces src/deploy/asl_model.task

    What a .task file actually is:
    - It is a ZIP archive (you can open it with any zip tool to inspect it)
    - It contains your .tflite model file
    - It contains a labels.txt file listing one class per line
    - It contains a metadata.json file describing the model inputs/outputs
      so that any tool reading the .task file knows what the model expects

    This format is compatible with tools that read .task files generically.
    Note: This is NOT the same as the full MediaPipe gesture_recognizer.task
    bundle (which also embeds palm detection and landmark models inside it).
    That full bundle can only be produced by MediaPipe Model Maker on Linux.
    This .task file is for standalone use with your custom inference code,
    or for use with MediaPipe Studio for testing/visualization.
    """
    TASK_OUT = DEPLOY_DIR / "asl_model.task"

    # Build a metadata dict that describes the model clearly
    metadata = {
        "name": "ASL Gesture Classifier",
        "description": (
            "Classifies ASL hand gestures from 63 normalized landmark floats "
            "(21 landmarks x xyz, relative to wrist). "
            "Input shape: [1, 63]. Output shape: [1, n_classes]."
        ),
        "version": "1.0",
        "author": "ASL Interpreter Project",
        "input": {
            "name": "landmarks",
            "description": (
                "63 float32 values: 21 hand landmarks each with "
                "x, y, z normalized relative to the wrist position."
            ),
            "shape": [1, 63],
            "dtype": "float32"
        },
        "output": {
            "name": "gesture_probabilities",
            "description": "Softmax probability for each gesture class.",
            "shape": [1, len(label_names)],
            "dtype": "float32"
        },
        "labels": label_names,
        "label_count": len(label_names),
        "training_info": {
            "landmark_normalization": "relative to wrist (landmark 0)",
            "landmarks_per_hand": 21,
            "coords_per_landmark": 3
        }
    }

    # Write the .task file as a ZIP containing three files:
    # 1. model.tflite  — the actual model weights
    # 2. labels.txt    — one label per line, index matches model output index
    # 3. metadata.json — human + machine readable description of the model
    with zipfile.ZipFile(TASK_OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # The tflite model itself
        zf.writestr("model.tflite", tflite_bytes)

        # Labels file — one per line, index = class index in model output
        zf.writestr("labels.txt", "\n".join(label_names))

        # Metadata JSON
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))

    log(f"Task file saved → {TASK_OUT}")
    log(f"  Contents of {TASK_OUT.name}:")
    log(f"    model.tflite  ({len(tflite_bytes):,} bytes)")
    log(f"    labels.txt    ({len(label_names)} classes)")
    log(f"    metadata.json (model description)")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log("=== ASL Model Training Started ===")
    try:
        X, y, label_names = collect_dataset()
        clf, X_train, y_train = train_classifier(X, y, label_names)
        tflite_bytes = export_tflite(clf, X_train, y_train, label_names)
        export_task(tflite_bytes, label_names)
        log("=== Training Complete ===")
        log(f"Output files in {DEPLOY_DIR}:")
        log(f"  asl_model.tflite  — load with tf.lite.Interpreter (your current code)")
        log(f"  asl_model.task    — ZIP bundle with model + labels + metadata")
        log(f"  labels.txt        — plain text labels")
    except Exception as e:
        log(f"[ERROR] Training failed: {e}")
        raise