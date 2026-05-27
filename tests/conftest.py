from collections.abc import Iterator
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from digit_api.config import Settings
from digit_api.main import create_app


class FakeClassifier:
    active_version = "test-version"

    def predict(self, image_array: np.ndarray, metadata):
        assert image_array.shape == (28, 28)
        selected_version = metadata.model_version or self.active_version
        probabilities = {str(index): 0.01 for index in range(10)}
        probabilities["7"] = 0.91
        return 7, 0.91, probabilities, selected_version


@pytest.fixture()
def png_image_bytes() -> bytes:
    image = Image.new("L", (28, 28), color=255)
    for x in range(10, 18):
        for y in range(4, 24):
            image.putpixel((x, y), 0)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture()
def client(tmp_path) -> Iterator[TestClient]:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        (
            '{"active_version":"test-version","models":{"test-version":{'
            '"image_model_path":"missing-image.pth",'
            '"classifier_path":"missing-classifier.pth",'
            '"metadata_encoder_path":"missing-encoder.joblib",'
            '"metadata_dim":4}}}'
        ),
        encoding="utf-8",
    )
    app = create_app(Settings(model_registry_path=registry_path))
    app.state.classifier = FakeClassifier()
    with TestClient(app) as test_client:
        yield test_client
