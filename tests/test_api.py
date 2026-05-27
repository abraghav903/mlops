import json


def test_health_returns_active_model_version(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "active_model_version": "test-version"}


def test_predict_accepts_image_and_metadata(client, png_image_bytes):
    response = client.post(
        "/predict",
        files={"image": ("digit.png", png_image_bytes, "image/png")},
        data={
            "metadata": json.dumps(
                {
                    "request_id": "req-123",
                    "source": "pytest",
                    "pen_pressure": 1.0,
                    "writer_age": 35,
                    "handedness": "right",
                    "model_version": "champion",
                    "tags": {"team": "mlops"},
                }
            )
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["prediction"] == 7
    assert body["confidence"] == 0.91
    assert body["model_version"] == "champion"
    assert body["metadata"]["request_id"] == "req-123"


def test_predict_rejects_invalid_metadata(client, png_image_bytes):
    response = client.post(
        "/predict",
        files={"image": ("digit.png", png_image_bytes, "image/png")},
        data={"metadata": "{not-json"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "metadata must be valid JSON"


def test_predict_rejects_non_image_upload(client):
    response = client.post(
        "/predict",
        files={"image": ("digit.txt", b"hello", "text/plain")},
        data={
            "metadata": json.dumps(
                {"pen_pressure": 1.0, "writer_age": 35, "handedness": "right"}
            )
        },
    )

    assert response.status_code == 415


def test_metrics_endpoint_exposes_prometheus_metrics(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "digit_api_requests_total" in response.text
