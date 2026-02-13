import os
import tomli
import toml
from pathlib import Path
from core.settings import Settings
from config.loader import loadSettings
from config.config import DEFAULT_CONFIG
CONFIG_DIR = Path(__file__).parent
FILENAME = "config.dev.toml"
CONFIG_PATH = CONFIG_DIR / FILENAME
def loadDefaultSettings():
    with open(CONFIG_PATH, "w") as f:
        toml.dump(DEFAULT_CONFIG, f)
    return loadSettings()