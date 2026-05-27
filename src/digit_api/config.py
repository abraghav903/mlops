import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "nxp-digit-inference"
    model_registry_path: Path = Field(default=Path("models/registry.json"))
    max_upload_bytes: int = 2_000_000
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", "nxp-digit-inference"),
        model_registry_path=Path(os.getenv("MODEL_REGISTRY_PATH", "models/registry.json")),
        max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", "2000000")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
