.PHONY: install train run test lint docker-build

install:
	python -m pip install -r requirements-dev.txt

train:
	python scripts/train_model.py --version v1 --output-dir models

run:
	uvicorn digit_api.main:app --app-dir src --reload --host 0.0.0.0 --port 8000

test:
	pytest

lint:
	ruff check .

docker-build:
	docker build -t nxp-digit-inference:latest .
