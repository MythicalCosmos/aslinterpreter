import os
import tomli
import toml
from pathlib import Path
import tempfile
from core.settings import Settings
from config.config import DEFAULT_CONFIG
CONFIG_DIR = Path(__file__).parent
FILENAME = "config.dev.toml"
CONFIG_PATH = CONFIG_DIR / FILENAME


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

    @classmethod
    def _write(cls, config):
        fd, temp_path = tempfile.mkstemp()
        with open(fd, "w", encoding="utf-8") as f:
            toml.dump(config, f)
        os.replace(temp_path, CONFIG_PATH)

    @classmethod
    def getConfig(cls):
        return cls._read()

    @classmethod
    def setEnvToken(cls, token: str) -> None:
        """
        Write the HuggingFace token to the [env] section.
        This is the only allowed write to the env section.
        The token is sensitive — don't log it.
        """
        config = cls._read()
        config["env"]["hf_token"] = token
        cls._write(config)

    @classmethod
    def update(cls, section: str, key: str, value):
        config = cls._read()
        if section not in config:
            raise KeyError(f"Section '{section}' not found")
        if key not in config[section]:
            raise KeyError(f"Key '{key}' not in [{section}]")
        if section == "env":
            raise PermissionError("env section is read-only")
        config[section][key] = value
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