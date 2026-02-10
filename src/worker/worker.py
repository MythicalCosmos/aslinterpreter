from pathlib import Path
import json
from mediapipe_model_maker.python.vision import gesture_recognizer as mp
import time

SHARED = Path(__file__).parent.parent / "shared"
LOG_FILE = Path(__file__).parent.parent / "shared/logs/worker.log.json"
LOG_FILE.parent.mkdir(exist_ok=True)
JOB_FILE = Path(__file__).parent.parent / "shared/job.json"

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

data = mp.gesture_recognizer.Dataset.from_folder(dirname=DATASET_PATH, hparams=mp.gesture_recognizer.HandDataPreprocessingParams())
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)
hparams = mp.gesture_recognizer.HParams(export_dir=EXPORT_PATH / "Exported")
options = mp.gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
model = mp.gesture_recognizer.GestureRecognizer.create(train_data=train_data, validation_data=validation_data, options=options)
loss, acc = model.evaluate(test_data, batch_size=1)
log(f"Test loss:{loss}, Test accuracy:{acc}")
model.export_model()
print("Exporting Model")
log("Exporting Model")
hparams = mp.gesture_recognizer.HParams(learning_rate=0.003, export_dir=EXPORT_PATH / "Final Export")
model_options = mp.gesture_recognizer.ModelOptions(dropout_rate=0.2)
options = mp.gesture_recognizer.GestureRecognizerOptions(model_options=model_options, hparams=hparams)
model_2 = mp.gesture_recognizer.GestureRecognizer.create(train_data=train_data, validation_data=validation_data, options=options)
loss, accuracy = model_2.evaluate(test_data)
log(f"Test loss:{loss}, Test accuracy:{accuracy}")
print("Model Exported Successfully")
log("Model Exported Successfully")