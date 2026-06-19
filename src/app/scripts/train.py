import os
import sys
import argparse
import numpy as np
import cv2
import json
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── CLI Arguments ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="ASL gesture model trainer")
parser.add_argument("--dataset", default=None, help="Gesture image dataset folder")
parser.add_argument("--output",  default=None, help="Deploy directory for model output")
parser.add_argument("--logfile", default=None, help="Training log path (polled by UI)")
parser.add_argument("--augment", action="store_true", default=True, help="Enable data augmentation")
parser.add_argument("--model",   default="ensemble", choices=["rf", "svm", "ensemble"],
                    help="Classifier type: rf=RandomForest, svm=SVM, ensemble=RF+SVM+GB")
args, _ = parser.parse_known_args()

# ── Path Resolution ────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent        # src/app/scripts/
_REPO_ROOT  = _SCRIPT_DIR.parent.parent.parent       # src/

DATASET_PATH = Path(args.dataset) if args.dataset else _REPO_ROOT / "shared/datasets"
DEPLOY_DIR   = Path(args.output)  if args.output  else _REPO_ROOT / "deploy"
LOG_OUT      = Path(args.logfile) if args.logfile  else _REPO_ROOT / "shared/logs/training_log.txt"

LANDMARK_MODEL = DEPLOY_DIR / "hand_landmarker.task"
MODEL_OUT      = DEPLOY_DIR / "asl_model.pkl"
LABELS_OUT     = DEPLOY_DIR / "labels.txt"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# ── Logging ────────────────────────────────────────────────────────────────────
def log(msg: str, level: str = "INFO") -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] [{level}] {msg}"
    print(line, flush=True)
    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_OUT, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def log_section(title: str) -> None:
    bar = "=" * 60
    log(f"\n{bar}")
    log(f"  {title}")
    log(f"{bar}")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Landmark Extraction
# ══════════════════════════════════════════════════════════════════════════════

def build_landmarker() -> mp_vision.HandLandmarker:
    """Create a MediaPipe HandLandmarker using the Tasks API (correct for ≥0.10)."""
    if not LANDMARK_MODEL.exists():
        raise FileNotFoundError(
            f"hand_landmarker.task not found at {LANDMARK_MODEL}\n"
            f"Run: python scripts/setup.py"
        )
    base_options = mp_python.BaseOptions(model_asset_path=str(LANDMARK_MODEL))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,   # lower threshold = more training samples
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def extract_landmarks(image_path: Path, landmarker: mp_vision.HandLandmarker) -> np.ndarray | None:
    """
    Extract a wrist-normalized 63-float landmark vector from one image.
    Returns None if no hand is detected.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    lm = result.hand_landmarks[0]   # first hand
    wrist = lm[0]

    # Wrist-normalize so position in frame doesn't affect the feature vector
    vec = np.array(
        [[l.x - wrist.x, l.y - wrist.y, l.z - wrist.z] for l in lm],
        dtype=np.float32
    ).flatten()   # shape: (63,)

    return vec


def augment_vector(vec: np.ndarray) -> list[np.ndarray]:
    """
    Generate augmented variants of one landmark vector.
    Augmentation happens in landmark space (not image space) — fast and clean.

    Augmentations applied:
      - Small Gaussian noise (simulates sensor jitter)
      - Scale variation ±10% (simulates distance from camera)
      - Minor axis translation (simulates hand position variation)

    Returns a list of augmented vectors (original NOT included — caller adds it).
    """
    augmented = []
    rng = np.random.default_rng()

    # Noise variants (3)
    for sigma in [0.005, 0.01, 0.015]:
        augmented.append(vec + rng.normal(0, sigma, vec.shape).astype(np.float32))

    # Scale variants (4)
    for scale in [0.90, 0.95, 1.05, 1.10]:
        augmented.append((vec * scale).astype(np.float32))

    # Combined noise + scale (2)
    for scale, sigma in [(0.95, 0.008), (1.05, 0.008)]:
        noisy = vec + rng.normal(0, sigma, vec.shape).astype(np.float32)
        augmented.append((noisy * scale).astype(np.float32))

    return augmented   # 9 augmented variants per original sample


def collect_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Walk DATASET_PATH, extract landmarks from every image, apply augmentation.
    Returns X (n_samples, 63), y (n_samples,), label_names.
    """
    if not DATASET_PATH.exists():
        raise RuntimeError(f"Dataset folder not found: {DATASET_PATH}")

    label_names = sorted([
        d.name for d in DATASET_PATH.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    if not label_names:
        raise RuntimeError(f"No gesture subfolders found in {DATASET_PATH}")
    if len(label_names) < 2:
        raise RuntimeError(
            f"Need at least 2 gesture classes to train — only {len(label_names)} found ({label_names}). "
            "Add a second gesture in the Model Maker tab."
        )

    log_section("Dataset Collection")
    log(f"Dataset path : {DATASET_PATH}")
    log(f"Found labels : {label_names}")
    log(f"Augmentation : {'ON - 9 variants per sample' if args.augment else 'OFF'}")

    X, y = [], []
    total_images = 0
    total_detected = 0

    landmarker = build_landmarker()
    log("HandLandmarker ready\n")

    for label_idx, label in enumerate(label_names):
        folder = DATASET_PATH / label
        images = [f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        good = 0

        for img_path in images:
            vec = extract_landmarks(img_path, landmarker)
            if vec is not None:
                X.append(vec)
                y.append(label_idx)
                good += 1
                if args.augment:
                    for aug_vec in augment_vector(vec):
                        X.append(aug_vec)
                        y.append(label_idx)

        samples_after_aug = good * (10 if args.augment else 1)
        status = "[OK]" if good >= 20 else ("[WARN]" if good >= 5 else "[FAIL]")
        log(f"  {status} {label:>4}  raw={good}/{len(images)}  "
            f"after_aug={samples_after_aug}")
        total_images += len(images)
        total_detected += good

    log(f"\nTotal images scanned : {total_images}")
    log(f"Landmarks detected   : {total_detected}")
    log(f"Training samples     : {len(X)} (with augmentation)")

    if len(X) == 0:
        raise RuntimeError("No landmarks could be extracted. Check your images and hand_landmarker.task.")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), label_names


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Model Training
# ══════════════════════════════════════════════════════════════════════════════

def build_classifier(model_type: str):
    """Build and return the selected classifier type."""

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",    # handles unequal class sizes gracefully
    )

    if model_type == "rf":
        return rf

    if model_type == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                           probability=True, random_state=42))
        ])

    # Ensemble: RandomForest + SVM + GradientBoosting, soft voting
    # This consistently outperforms any single model on landmark data
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                       probability=True, random_state=42))
    ])
    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    )
    return VotingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("gb", gb)],
        voting="soft",
        n_jobs=-1,
    )


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    model_type: str = "ensemble",
):
    """
    Train the classifier with stratified train/test split.
    Prints per-class accuracy and confusion matrix summary.
    """
    log_section("Model Training")
    log(f"Model type   : {model_type}")
    log(f"Samples      : {len(X)}")
    log(f"Features     : {X.shape[1]} (63 landmark coords)")
    log(f"Classes      : {len(label_names)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    log(f"Train / test : {len(X_train)} / {len(X_test)}")
    log("Training...\n")

    clf = build_classifier(model_type)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    overall_acc = accuracy_score(y_test, y_pred)

    log(f"Overall accuracy : {overall_acc:.2%}")
    log("\nPer-class accuracy:")

    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    for label in label_names:
        if label in report:
            r = report[label]
            icon = "[OK]" if r["f1-score"] >= 0.90 else ("[WARN]" if r["f1-score"] >= 0.75 else "[FAIL]")
            log(f"  {icon} {label:>4}  precision={r['precision']:.2f}  "
                f"recall={r['recall']:.2f}  f1={r['f1-score']:.2f}  "
                f"support={int(r['support'])}")

    # Cross-validation for more reliable estimate
    log("\nRunning 5-fold cross-validation...")
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    log(f"CV accuracy : {cv_scores.mean():.2%} +/- {cv_scores.std():.2%}")

    # Confusion matrix summary — only show worst pairs
    cm = confusion_matrix(y_test, y_pred)
    log("\nTop confusion pairs (most confused):")
    confusion_pairs = []
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            if i != j and cm[i][j] > 0:
                confusion_pairs.append((cm[i][j], label_names[i], label_names[j]))
    confusion_pairs.sort(reverse=True)
    for count, true_label, pred_label in confusion_pairs[:5]:
        log(f"  {true_label} -> {pred_label} : {count} times")

    return clf, overall_acc


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Export
# ══════════════════════════════════════════════════════════════════════════════

def export_model(clf, label_names: list[str], accuracy: float) -> None:
    """Save model, labels, and metadata to deploy directory."""
    log_section("Exporting Model")
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, MODEL_OUT, compress=3)
    log(f"Model saved  : {MODEL_OUT}")

    LABELS_OUT.write_text("\n".join(label_names), encoding="utf-8")
    log(f"Labels saved : {LABELS_OUT}")

    # Save metadata alongside the model for the UI to display
    meta = {
        "trained_at":    datetime.now().isoformat(),
        "labels":        label_names,
        "num_classes":   len(label_names),
        "accuracy":      round(accuracy, 4),
        "model_type":    args.model,
        "augmented":     args.augment,
        "dataset_path":  str(DATASET_PATH),
    }
    meta_path = DEPLOY_DIR / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"Metadata     : {meta_path}")
    log(f"\nFinal accuracy: {accuracy:.2%}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Clear log from previous runs
    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    LOG_OUT.write_text("", encoding="utf-8")

    log_section("ASL Model Training Started")
    log(f"Dataset   : {DATASET_PATH}")
    log(f"Deploy    : {DEPLOY_DIR}")
    log(f"Log       : {LOG_OUT}")

    try:
        X, y, label_names = collect_dataset()
        clf, accuracy = train_classifier(X, y, label_names, model_type=args.model)
        export_model(clf, label_names, accuracy)
        log_section("Training Complete [OK]")
        log(f"ACCURACY:{accuracy:.4f}")   # parsed by UI for accuracy display
    except Exception as e:
        log(f"Training failed: {e}", level="ERROR")
        import traceback
        log(traceback.format_exc(), level="ERROR")
        sys.exit(1)