import toml
import tomli
from pathlib import Path
from core.settings import Settings
from config.config import DEFAULT_CONFIG

CONFIG_DIR  = Path(__file__).parent
CONFIG_PATH = CONFIG_DIR / "config.dev.toml"


def loadSettings() -> Settings:
    try:
        # tomli for reading (binary, spec-compliant), toml for writing
        with open(CONFIG_PATH, "rb") as f:
            data = tomli.load(f)
        if not data:
            raise ValueError("Empty config file")
        return Settings(**data)
    except Exception:
        # Fallback: write defaults and return them
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            toml.dump(DEFAULT_CONFIG, f)
        return Settings(**DEFAULT_CONFIG)