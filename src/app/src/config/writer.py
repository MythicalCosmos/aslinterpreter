import os
import tomli
import toml
from pathlib import Path
from core.settings import Settings
from config.config import DEFAULT_CONFIG
CONFIG_DIR = Path(__file__).parent
FILENAME = "config.dev.toml"
CONFIG_PATH = CONFIG_DIR / FILENAME


@staticmethod
def _merge(default, user):
    result = default.copy()
    for k, v in user.items():
        if isinstance(v, dict) and k in result:
            result[k] = _merge(result[k], v)
        else:
            result[k] = v
    return result
    

class ConfigAPI:
    @staticmethod
    def _read():
        with CONFIG_PATH.open("rb") as f:
            userConfig = tomli.load(f)

        return _merge(DEFAULT_CONFIG, userConfig)

    @staticmethod
    def _write(config: dict):
        with CONFIG_PATH.open("wb") as f:
            toml.dump(config, f)

    @classmethod
    def getConfig(cls):
        return cls._read()

    @classmethod
    def update(cls, section: str, key: str, value):
        config = cls._read()
        if section not in config:
            raise KeyError(f"Section '{section}' not found")
        if key not in config[section]:
            raise KeyError(f"Key '{key}' not in [{section}]")
        config[section][key] = value
        if section == "env":
            raise PermissionError("env section is read-only")
        cls._write(config)
        return config

    @classmethod
    def bulkUpdate(cls, updates: dict):
        config = cls._read()
        for section, values in updates.items():
            if section not in config:
                raise KeyError(f"Section '{section}' not found")
            for key, value in values.items():
                if key not in config[section]:
                    raise KeyError(f"{key} not in [{section}]")
                config[section][key] = value
        cls._write(config)
        return config