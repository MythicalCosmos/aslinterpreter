import os
import tomli
from pathlib import Path
from core.settings import Settings
CONFIG_DIR = Path(__file__).parent
def load_settings() -> Settings:
    filename = "config.dev.toml"
    configPath = CONFIG_DIR / filename
    if not configPath.exists():
        raise RuntimeError(f"Config file not found: {configPath}")
    with open(configPath, "rb") as f:
        data = tomli.load(f)
    return Settings(**data)