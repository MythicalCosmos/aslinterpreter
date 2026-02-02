from pydantic import BaseModel
from typing import Optional

class AppSettings(BaseModel):
    name: str
    environment: str
    width: int
    height: int

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

class envSettings(BaseModel):
    hf_token: str

class Settings(BaseModel):
    app: AppSettings
    api: ApiSettings
    gestures: GestureSettings
    settings: Usersettings
    env: envSettings
