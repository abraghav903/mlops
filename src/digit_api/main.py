import json
import logging
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError

from digit_api.config import Settings, get_settings
from digit_api.image_processing import load_image, preprocess_digit_image
from digit_api.logging_config import configure_logging
from digit_api.metrics import PREDICTION_COUNT, REQUEST_COUNT, REQUEST_LATENCY
from digit_api.model import DigitClassifier, ModelRegistry
from digit_api.schemas import HealthResponse, PredictionMetadata, PredictionResponse

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="NXP Digit Prediction API",
        version="0.1.0",
        description="REST API for handwritten digit classification.",
    )

    registry = ModelRegistry(settings.model_registry_path)
    classifier = DigitClassifier(registry)
    app.state.classifier = classifier
    app.state.settings = settings

    @app.get("/health", response_model=HealthResponse, tags=["observability"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", active_model_version=classifier.active_version)

    @app.get("/metrics", tags=["observability"])
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict", response_model=PredictionResponse, tags=["inference"])
    async def predict(
        image: Annotated[UploadFile, File()],
        metadata: Annotated[str, Form()] = "{}",
    ) -> PredictionResponse:
        started = perf_counter()
        try:
            active_classifier: DigitClassifier = app.state.classifier
            parsed_metadata = _parse_metadata(metadata)
            content = await image.read()
            _validate_upload(image, content, app.state.settings.max_upload_bytes)

            pil_image = load_image(content)
            image_array = preprocess_digit_image(pil_image)
            prediction, confidence, probabilities, model_version = active_classifier.predict(
                image_array,
                parsed_metadata,
            )

            PREDICTION_COUNT.labels(digit=str(prediction), model_version=model_version).inc()
            REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
            logger.info(
                "prediction_completed",
                extra={
                    "prediction": prediction,
                    "confidence": confidence,
                    "model_version": model_version,
                    "request_id": parsed_metadata.request_id,
                },
            )
            return PredictionResponse(
                prediction=prediction,
                confidence=confidence,
                probabilities=probabilities,
                model_version=model_version,
                metadata=parsed_metadata.model_dump(exclude_none=True),
            )
        except HTTPException:
            REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
            raise
        except ValueError as exc:
            REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("prediction_failed")
            REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction failed",
            ) from exc
        finally:
            REQUEST_LATENCY.labels(endpoint="/predict").observe(perf_counter() - started)

    return app


def _parse_metadata(raw_metadata: str) -> PredictionMetadata:
    try:
        payload = json.loads(raw_metadata or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail="metadata must be valid JSON",
        ) from exc

    try:
        return PredictionMetadata.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(),
        ) from exc


def _validate_upload(image: UploadFile, content: bytes, max_upload_bytes: int) -> None:
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="image file is empty")
    if len(content) > max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="image file is too large",
        )
    if image.content_type not in {"image/png", "image/jpeg", "image/jpg", "image/bmp"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="image must be PNG, JPEG, or BMP",
        )


try:
    app = create_app()
except FileNotFoundError as exc:
    logger.warning("API started without a model registry: %s", exc)
    app = FastAPI(title="NXP Digit Prediction API", version="0.1.0")

    @app.get("/health", response_model=HealthResponse, tags=["observability"])
    def health_unavailable() -> HealthResponse:
        return HealthResponse(status="model_unavailable", active_model_version=None)
