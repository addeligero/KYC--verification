import os
from functools import lru_cache
from pydantic import BaseModel


class Settings(BaseModel):
    # Face matching thresholds
    FACE_PASS_THRESHOLD: float = float(os.getenv("FACE_PASS_THRESHOLD", "0.35"))  # cosine sim threshold
    LIVENESS_PASS_THRESHOLD: float = float(os.getenv("LIVENESS_PASS_THRESHOLD", "0.35"))

    # Sanctions
    OPEN_SANCTIONS_API_KEY: str | None = os.getenv("OPEN_SANCTIONS_API_KEY") or None
    SANCTIONS_TOPK: int = int(os.getenv("SANCTIONS_TOPK", "5"))
    SANCTIONS_FLAG_THRESHOLD: float = float(os.getenv("SANCTIONS_FLAG_THRESHOLD", "0.85"))


@lru_cache
def get_settings() -> Settings:
    return Settings()