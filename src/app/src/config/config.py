"""Configuration module for config."""

from pathlib import Path
import tomli

DEFAULT_CONFIG = {
    "version": {
        "version": "1.0.0"
    },
    "app": {
        "name": "Asl to English Translator",
        "environment": "development",
        "fullscreen_mode": "Windowed",
        "width": 1280,
        "height": 800,
        "pos_x": 100,
        "pos_y": 100,
        "monitor": 0,
        "dpi_scaling": True,
        "camera": 0,
        "log_level": 1
    },
    "api": {
        "base_url": "http://localhost:8080"
    },
    "gestures": {
        "gesture_model": "testing"
    },
    "settings": {
        "examples": 5,
        "sam_rate": 16000,
        "init_chunk_der": 8.0,
        "min_chunk_der": 2.0,
        "chunk_dec": 1.0,
        "lines": False,
        "confidence_threshold": 0.75,
        "autocorrect": True,
        "autocorrect_threshold": 0.75,
        "word_gap": 3.0,
        "preview_toggle": True,
        "confidence_toggle": True
    },
    "env": {
        "hf_token": ""
    }
}
