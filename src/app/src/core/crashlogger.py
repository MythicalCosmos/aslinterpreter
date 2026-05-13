import sys, json, traceback
from pathlib import Path
from datetime import datetime

CRASH_LOG = Path(__file__).parent.parent.parent.parent / "logs/crashes.jsonl"

def installCrashHandler():
    def handler(exc_type, exc_value, exc_tb):
        entry = {
            "time": datetime.now().isoformat(),
            "type": exc_type.__name__,
            "message": str(exc_value),
            "traceback": traceback.format_tb(exc_tb)
        }
        CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(CRASH_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
        sys.__excepthook__(exc_type, exc_value, exc_tb)