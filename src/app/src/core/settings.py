from pydantic import BaseModel
from typing import Optional

class VersionSettings(BaseModel):
    version: str

class AppSettings(BaseModel):
    name: str
    environment: str
    width: int
    height: int
    log_level: int

class ApiSettings(BaseModel):
    base_url: str

class GestureSettings(BaseModel):
    gesture_model: str

class Usersettings(BaseModel):
    examples: int
    sam_rate: int
    init_chunk_der: float
    min_chunk_der: float
    chunk_dec: float
    lines: bool
    confidence_threshold: float
    autocorrect: bool
    autocorrect_threshold: float
    word_gap: float
    preview_toggle: bool
    confidence_toggle: bool

class envSettings(BaseModel):
    hf_token: str

class Settings(BaseModel):
    version: VersionSettings
    app: AppSettings
    api: ApiSettings
    gestures: GestureSettings
    settings: Usersettings
    env: envSettings