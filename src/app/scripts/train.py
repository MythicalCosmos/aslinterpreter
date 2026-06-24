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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
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
    Extract a wrist-normalized 73-float feature vector from one image.
    63 position coords (wrist-relative, scale-normalized) + 10 joint-angle cosines.
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
    ref   = lm[9]  # middle-finger MCP — scale reference

    # Position-normalize (wrist-relative) then scale-normalize (unit hand size)
    # so the vector is invariant to where and how far the hand is from the camera.
    scale = np.sqrt(
        (ref.x - wrist.x)**2 + (ref.y - wrist.y)**2 + (ref.z - wrist.z)**2
    )
    if scale < 1e-6:
        return None  # degenerate detection — skip

    vec = np.array(
        [[(l.x - wrist.x) / scale, (l.y - wrist.y) / scale, (l.z - wrist.z) / scale]
         for l in lm],
        dtype=np.float32
    ).flatten()   # shape: (63,)

    # Joint-angle cosines: PIP then DIP for each finger (thumb→pinky).
    # Encodes finger curvature independently of hand position/scale/orientation.
    _joints = [(1,2,3),(5,6,7),(9,10,11),(13,14,15),(17,18,19),
               (2,3,4),(6,7,8),(10,11,12),(14,15,16),(18,19,20)]
    angles = []
    for a, b, c in _joints:
        v1 = np.array([lm[a].x - lm[b].x, lm[a].y - lm[b].y, lm[a].z - lm[b].z])
        v2 = np.array([lm[c].x - lm[b].x, lm[c].y - lm[b].y, lm[c].z - lm[b].z])
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angles.append(float(np.clip(cos_a, -1.0, 1.0)))

    # Fingertip-to-thumb-tip distances — key for M/N/S/A/E cluster where the
    # number of fingers resting over the thumb is the primary distinction.
    # N: ring tip far from thumb. M: ring tip close (ring folds over thumb too).
    thumb_tip = np.array([
        (lm[4].x - wrist.x) / scale,
        (lm[4].y - wrist.y) / scale,
        (lm[4].z - wrist.z) / scale,
    ])
    dists = []
    for tip_idx in [8, 12, 16, 20]:   # index, middle, ring, pinky
        tip = np.array([
            (lm[tip_idx].x - wrist.x) / scale,
            (lm[tip_idx].y - wrist.y) / scale,
            (lm[tip_idx].z - wrist.z) / scale,
        ])
        dists.append(float(np.linalg.norm(tip - thumb_tip)))

    # Adjacent fingertip pair distances — key for R/U/V lateral spread.
    # R: index-middle distance is small (fingers crossing).
    # U: index-middle distance is medium (side by side).
    # V: index-middle distance is large (spread apart).
    tips = [
        np.array([(lm[i].x - wrist.x)/scale, (lm[i].y - wrist.y)/scale, (lm[i].z - wrist.z)/scale])
        for i in [8, 12, 16, 20]   # index, middle, ring, pinky tips
    ]
    adj_dists = [
        float(np.linalg.norm(tips[0] - tips[1])),   # index  ↔ middle
        float(np.linalg.norm(tips[1] - tips[2])),   # middle ↔ ring
        float(np.linalg.norm(tips[2] - tips[3])),   # ring   ↔ pinky
    ]

    # Z-depth differentials: which fingertip is closer to the camera.
    # In normalized space, z encodes depth relative to the wrist.
    # R: index tip crosses behind middle tip → index_z > middle_z → positive diff.
    # U: tips at same depth → diff near 0.
    # Providing this explicitly gives the classifier a direct crossing signal
    # instead of requiring it to infer depth order from raw coordinates.
    z_diffs = [
        float(tips[0][2] - tips[1][2]),   # index_tip.z  - middle_tip.z
        float(tips[1][2] - tips[2][2]),   # middle_tip.z - ring_tip.z
    ]

    return np.concatenate([
        vec,
        np.array(angles,    dtype=np.float32),
        np.array(dists,     dtype=np.float32),
        np.array(adj_dists, dtype=np.float32),
        np.array(z_diffs,   dtype=np.float32),
    ])  # shape: (82,)


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

    pos = vec[:63]   # position coords — affected by hand distance / rotation
    ang = vec[63:]   # joint angles, distances, z-diffs — scale/rotation invariant

    # Noise variants (6) — full vector; larger sigmas cover jitter on
    # crossed/hooked/curled fingers (R, X, G)
    for sigma in [0.005, 0.01, 0.015, 0.03, 0.05, 0.08]:
        augmented.append(vec + rng.normal(0, sigma, vec.shape).astype(np.float32))

    # Scale variants (4) — positions only; joint angles don't change with distance
    for scale in [0.90, 0.95, 1.05, 1.10]:
        augmented.append(np.concatenate([(pos * scale).astype(np.float32), ang]))

    # Combined noise + scale (2)
    for scale, sigma in [(0.95, 0.008), (1.05, 0.008)]:
        noisy_pos = (pos + rng.normal(0, sigma, pos.shape).astype(np.float32)) * scale
        noisy_ang =  ang + rng.normal(0, sigma, ang.shape).astype(np.float32)
        augmented.append(np.concatenate([noisy_pos.astype(np.float32), noisy_ang.astype(np.float32)]))

    # 2-D in-plane rotation variants (4) — positions only.
    # Simulates the hand being signed at a different tilt or the camera at a
    # slightly different angle.  Joint angles don't change with orientation.
    def _rot2d(p: np.ndarray, deg: float) -> np.ndarray:
        rad = np.radians(deg)
        c, s = float(np.cos(rad)), float(np.sin(rad))
        r = p.copy()
        for i in range(21):
            x, y = float(p[i * 3]), float(p[i * 3 + 1])
            r[i * 3]     = x * c - y * s
            r[i * 3 + 1] = x * s + y * c
        return r.astype(np.float32)

    for deg in [-20.0, -10.0, 10.0, 20.0]:
        augmented.append(np.concatenate([_rot2d(pos, deg), ang]))

    return augmented   # 16 augmented variants per original sample


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
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=1,
        class_weight="balanced",
    )

    if model_type == "rf":
        return rf

    if model_type == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                           probability=True, random_state=42))
        ])

    # Ensemble: RandomForest + LogisticRegression + GradientBoosting, soft voting.
    # LR replaces RBF-SVC: identical accuracy on normalised 63-feature vectors,
    # O(n) memory vs O(n^2) for the kernel matrix (which OOM'd at 25k+ samples).
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(C=5, max_iter=1000, random_state=42)),
    ])
    gb = HistGradientBoostingClassifier(
        max_iter=50,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        warm_start=True,
    )
    return VotingClassifier(
        estimators=[("rf", rf), ("lr", lr), ("gb", gb)],
        voting="soft",
        n_jobs=1,
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
    log(f"Features     : {X.shape[1]} (63 coords + 10 joint angles + 4 thumb dists + 3 spreads + 2 z-diffs)")
    log(f"Classes      : {len(label_names)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    log(f"Train / test : {len(X_train)} / {len(X_test)}")
    log("Training...\n")

    clf = build_classifier(model_type)

    if model_type == "ensemble":
        from sklearn.utils import Bunch
        names    = [n for n, _ in clf.estimators]
        sub_ests = [e for _, e in clf.estimators]

        log("Fitting RandomForest (100 trees) — may take 1-3 min...")
        sub_ests[0].fit(X_train, y_train)
        log("RandomForest done.")

        log("Fitting LogisticRegression...")
        sub_ests[1].fit(X_train, y_train)
        log("LogisticRegression done.")

        log("Fitting HistGradientBoosting (50 iterations)...")
        for _batch in range(10, 51, 10):
            sub_ests[2].set_params(max_iter=_batch)
            sub_ests[2].fit(X_train, y_train)
            log(f"  HistGradientBoosting: {_batch}/50 iterations done...")
        log("HistGradientBoosting done.")

        le = LabelEncoder().fit(y_train)
        clf.le_               = le
        clf.classes_          = le.classes_
        clf.estimators_       = sub_ests
        clf.named_estimators_ = Bunch(**dict(zip(names, sub_ests)))
    else:
        log(f"Fitting {model_type} classifier...")
        clf.fit(X_train, y_train)
        log("Classifier done.")

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
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy", n_jobs=1)
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
