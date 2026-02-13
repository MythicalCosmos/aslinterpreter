from pathlib import Path
import tomli

DEFAULT_CONFIG = {
    "version": {
        "version": "1.0.0"
    },
    "app": {
        "name": "Asl to English Translator",
        "enviornment": "development",
        "width": 1280,
        "height": 800,
        "log_level": 1,
    },
    "api": {
        "base_url": "http://localhost:8080",
    },
    "gestures": {
        "gesture_model": "testing",
    },
    "settings": {
        "examples": 5,
        "sam_rate": 16000,
        "init_chunk_der": 8.0,
        "min_chunk_der": 2.0,
        "chunk_dec": 1.0,
        "lines": "false",
        "confidence_threshold": 0.75,
        "autocorrect": "true",
        "autocorrect_threshold": 0.85,
        "word_gap": 1.0,
        "preview_toggle": "true",
        "confidence_toggle": "true",
    },
    "env": {
        "hf_token": "",
    }
}
