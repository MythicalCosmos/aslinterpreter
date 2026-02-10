import os
import tomli
from pathlib import Path
from core.settings import Settings
CONFIG_DIR = Path(__file__).parent
FILENAME = "config.dev.toml"
CONFIG_PATH = CONFIG_DIR / FILENAME
def loadSettings() -> Settings:
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Config file not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "rb") as f:
        data = tomli.load(f)
    return Settings(**data)