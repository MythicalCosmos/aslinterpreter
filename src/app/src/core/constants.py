from pathlib import Path
from config.loader import loadSettings

# ── Repository layout ─────────────────────────────────────────────────────
_SRC_FILE  = Path(__file__).resolve()         # core/constants.py
_CORE_DIR  = _SRC_FILE.parent                 # src/app/src/core/
_APP_SRC   = _CORE_DIR.parent                 # src/app/src/
_REPO_ROOT = _APP_SRC.parent.parent.parent    # src/  (repo root)

# ── Data paths ────────────────────────────────────────────────────────────
SHARED          = _REPO_ROOT / "shared"
DATASET_PATH    = SHARED / "datasets"
EXPORT_PATH     = SHARED / "exports"
JSON_FILE       = SHARED / "gestures.json"
LOG_DIR         = SHARED / "logs"
WORKER_LOG_PATH = LOG_DIR / "training_log.txt"

# ── Model paths ───────────────────────────────────────────────────────────
DEPLOY_DIR            = _REPO_ROOT / "src/deploy"
LANDMARK_MODEL_PATH   = DEPLOY_DIR / "hand_landmarker.task"  # MediaPipe binary
CLASSIFIER_MODEL_PATH = DEPLOY_DIR / "asl_model.pkl"         # sklearn classifier
LABELS_PATH           = DEPLOY_DIR / "labels.txt"
WORDLIST              = DEPLOY_DIR / "words.txt"

# ── Settings (loaded once at startup) ─────────────────────────────────────
SETTINGS               = loadSettings()
HF_TOKEN               = SETTINGS.env.hf_token
VERSION                = SETTINGS.version.version
CAMERA_INDEX           = SETTINGS.app.camera
NUM_EXAMPLES           = SETTINGS.settings.examples
SAMPLE_RATE            = SETTINGS.settings.sam_rate
INITIAL_CHUNK_DURATION = SETTINGS.settings.init_chunk_der
MIN_CHUNK_DURATION     = SETTINGS.settings.min_chunk_der
CHUNK_DECREMENT        = SETTINGS.settings.chunk_dec
CONFIDENCE_THRESHOLD   = SETTINGS.settings.confidence_threshold
WORD_GAP               = SETTINGS.settings.word_gap
AUTOCORRECT_TOGGLE     = SETTINGS.settings.autocorrect
AUTOCORRECT_THRESHOLD  = SETTINGS.settings.autocorrect_threshold
LOG_LEVEL              = SETTINGS.app.log_level

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

# ── Hand landmark connection pairs (for overlay drawing) ──────────────────
HAND_CONNECTIONS = [
    (0, 1), (0, 5), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]