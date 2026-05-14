import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setupLogging(log_dir: Path, level=logging.INFO):
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "app.log", maxBytes=5_000_000, backupCount=3
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    root.addHandler(logging.StreamHandler())   # keep console output too
    return root