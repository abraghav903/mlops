# NXP Digit Inference API

Productionized handwritten digit classification service for the NXP MLOps assignment. The project wraps the notebook's PyTorch CNN plus metadata inference flow in a FastAPI REST API with validation, logging, Prometheus metrics, health checks, tests, containerization, CI, and a simple model-version registry.

## Project Structure

```text
src/digit_api/          FastAPI app, preprocessing, model loading, metrics
scripts/train_model.py  Training/export script that writes the model registry
tests/                  API and preprocessing tests
infra/aws-ecs/          Terraform starter for AWS ECS/Fargate deployment
.github/workflows/     Pull request CI and tag release workflow
```

## Local Setup

Create a Python 3.11+ environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

Train and register the default model:

```bash
python scripts/train_model.py --version v1 --output-dir models
```

Run the API:

```bash
uvicorn digit_api.main:app --app-dir src --reload --host 0.0.0.0 --port 8000
```

Open:

- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Prometheus metrics: http://localhost:8000/metrics

## Prediction Request

The API accepts a multipart request with:

- `image`: PNG, JPEG, or BMP handwritten digit image
- `metadata`: JSON string with required `pen_pressure`, `writer_age`, and `handedness`; optional `request_id`, `source`, `model_version`, and `tags`

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@sample_digit.png;type=image/png" \
  -F 'metadata={"pen_pressure":1.0,"writer_age":35,"handedness":"right","request_id":"demo-1","source":"curl","tags":{"owner":"mlops"}}'
```

Response:

```json
{
  "prediction": 7,
  "confidence": 0.91,
  "probabilities": {"0": 0.01, "1": 0.02},
  "model_version": "v1",
  "metadata": {"pen_pressure": 1.0, "writer_age": 35, "handedness": "right", "request_id": "demo-1", "source": "curl"}
}
```

## Testing And Linting

```bash
ruff check .
pytest
```

## Container Build And Run

Build the image. The Docker build trains and registers `v1` so the image works out of the box.

```bash
docker build -t nxp-digit-inference:latest .
docker run --rm -p 8000:8000 nxp-digit-inference:latest
```

## Prometheus Monitoring

The API exposes Prometheus metrics at:

```text
http://localhost:8000/metrics
```

Run the API and Prometheus together:

```bash
docker compose up --build
```

Open Prometheus:

```text
http://localhost:9090
```

Useful queries:

```promql
digit_api_requests_total
digit_api_predictions_total
rate(digit_api_requests_total[5m])
histogram_quantile(0.95, rate(digit_api_request_latency_seconds_bucket[5m]))
```

Prometheus scrape configuration lives in `monitoring/prometheus.yml`.

## Model Versioning

Model artifacts are stored under `models/<version>/`. The active model is tracked in `models/registry.json`:

```json
{
  "active_version": "v1",
  "models": {
    "v1": {
      "image_model_path": "v1/image_model.pth",
      "classifier_path": "v1/final_classifier.pth",
      "metadata_encoder_path": "v1/metadata_encoder.joblib",
      "metadata_dim": 4,
      "metrics": {"training_accuracy": 0.96}
    }
  }
}
```

To add a new version:

```bash
python scripts/train_model.py --version v2 --output-dir models
```

Clients can request a specific version by setting `metadata.model_version`. If omitted, the active registry version is used.

## CI/CD

The GitHub Actions pull request workflow:

1. Installs dependencies
2. Runs `ruff check .`
3. Trains the model artifact
4. Runs `pytest`
5. Builds the Docker image

The release workflow builds and pushes a container to GitHub Container Registry when a tag matching `v*.*.*` is pushed.

## Cloud Deployment IaC

`infra/aws-ecs/` contains a Terraform starter for AWS ECS/Fargate:

```bash
cd infra/aws-ecs
terraform init
terraform plan \
  -var="container_image=<image-uri>" \
  -var="execution_role_arn=<ecs-task-execution-role-arn>"
```

It creates an ECR repository, CloudWatch log group, ECS cluster, and task definition. In a real environment, add VPC, subnets, security groups, load balancer, and ECS service resources based on the target AWS account network setup.
