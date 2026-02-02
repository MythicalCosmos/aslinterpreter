from pathlib import Path
import json
from mediapipe_model_maker.python.vision.gesture_recognizer import gesture_recognizer as mp
import time

SHARED = Path("/shared")
LOG_FILE = SHARED / "logs" / "worker.log"
LOG_FILE.parent.mkdir(exist_ok=True)
JOB_FILE = SHARED / "job.json"
if not JOB_FILE.exists():
    raise RuntimeError("job.json not found in /shared")
with open(JOB_FILE, "r") as f:
    JOB = json.load(f)
def log(msg, level="info"):
    entry = {
        "time": time.time(),
        "level": level,
        "message": msg
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
DATASET_PATH = SHARED / JOB["dataset"]
EXPORT_PATH = SHARED / JOB["export"]
log(f"Dataset: {DATASET_PATH}")
log(f"Export: {EXPORT_PATH}")
EXPORT_PATH.mkdir(parents=True, exist_ok=True)
data = mp.python.vision.gesture_recognizer.Dataset.from_folder(
    dirname=DATASET_PATH,
    hparams=mp.python.vision.gesture_recognizer.HandDataPreprocessingParams()
)
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)
log("Training initial model")
hparams = mp.python.vision.gesture_recognizer.HParams(export_dir=EXPORT_PATH / "Exported")
options = mp.python.vision.gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
model = mp.python.vision.gesture_recognizer.GestureRecognizer.create(train_data=train_data, validation_data=validation_data, options=options)
loss, acc = model.evaluate(test_data, batch_size=1)
log(f"Test loss: {loss}, Test accuracy: {acc}")
model.export_model()
log("Initial model exported")
log("Training final model")
hparams = mp.HParams(
    learning_rate=0.003,
    export_dir=EXPORT_PATH / "Final Export"
)
model_options = mp.python.vision.gesture_recognizer.ModelOptions(dropout_rate=0.2)
options = mp.python.vision.gesture_recognizer.GestureRecognizerOptions(
    model_options=model_options,
    hparams=hparams
)
model_2 = mp.python.vision.gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)
loss, accuracy = model_2.evaluate(test_data)
log(f"Final loss: {loss}, Final accuracy: {accuracy}")
# ---- RESULT FILE ----
with open(SHARED / "result.json", "w") as f:
    json.dump({
        "status": "ok",
        "loss": loss,
        "accuracy": accuracy,
        "export_path": str(EXPORT_PATH)
    }, f)
log("Model exported successfully")