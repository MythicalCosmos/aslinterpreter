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
    examples: str
    sam_rate: int
    init_chunk_der: bool
    min_chunk_der: bool
    chunk_dec: bool

class Settings(BaseModel):
    app: AppSettings
    api: ApiSettings
    gestures: GestureSettings
    settings: Usersettings
