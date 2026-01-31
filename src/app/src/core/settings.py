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

class Settings(BaseModel):
    app: AppSettings
    api: ApiSettings
    gestures: GestureSettings
    settings: Usersettings
