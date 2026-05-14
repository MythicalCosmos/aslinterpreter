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
import joblib

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
def export_sklearn(clf, label_names):
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, DEPLOY_DIR / "asl_model.pkl")
    (DEPLOY_DIR / "labels.txt").write_text("\n".join(label_names))
    log("Model saved as asl_model.pkl")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log("=== ASL Model Training Started ===")
    try:
        X, y, label_names = collect_dataset()
        clf, X_train, y_train = train_classifier(X, y, label_names)
        export_sklearn(clf, label_names)
        log("=== Training Complete ===")
        log(f"Output files in {DEPLOY_DIR}:")
        log(f"  asl_model.pkl  — load with joblib.load()")
        log(f"  labels.txt     — plain text labels")
    except Exception as e:
        log(f"[ERROR] Training failed: {e}")
        raise