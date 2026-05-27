from typing import Any

from pydantic import BaseModel, Field, field_validator


class PredictionMetadata(BaseModel):
    pen_pressure: float = Field(gt=0.0, le=5.0)
    writer_age: int = Field(ge=1, le=120)
    handedness: str
    request_id: str | None = Field(default=None, max_length=128)
    source: str | None = Field(default=None, max_length=128)
    model_version: str | None = Field(
        default=None,
        description="Optional model version override. Defaults to the active registry version.",
    )
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("handedness")
    @classmethod
    def validate_handedness(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"left", "right"}:
            raise ValueError("handedness must be either 'left' or 'right'")
        return normalized

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > 20:
            raise ValueError("metadata.tags can contain at most 20 entries")
        for key, item in value.items():
            if len(key) > 64 or len(item) > 256:
                raise ValueError("metadata tag keys must be <=64 chars and values <=256 chars")
        return value


class PredictionResponse(BaseModel):
    prediction: int = Field(ge=0, le=9)
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    model_version: str
    metadata: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    active_model_version: str | None
