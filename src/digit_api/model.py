import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from digit_api.schemas import PredictionMetadata


class CNNEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc(x)


class FinalClassifier(nn.Module):
    def __init__(
        self,
        image_feat_dim: int = 128,
        metadata_dim: int = 4,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(image_feat_dim + metadata_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, img_feat: torch.Tensor, meta_feat: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([img_feat, meta_feat], dim=1))


@dataclass(frozen=True)
class ModelInfo:
    version: str
    image_model_path: Path
    classifier_path: Path
    metadata_encoder_path: Path
    metadata_dim: int
    metrics: dict[str, float]


class ModelRegistry:
    def __init__(self, registry_path: Path) -> None:
        self.registry_path = registry_path
        self._registry = self._load_registry()

    def _load_registry(self) -> dict:
        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Model registry not found at {self.registry_path}. "
                "Run `python scripts/train_model.py` before starting the API."
            )
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    @property
    def active_version(self) -> str:
        return self._registry["active_version"]

    def resolve(self, version: str | None = None) -> ModelInfo:
        requested_version = version or self.active_version
        models = self._registry.get("models", {})
        if requested_version not in models:
            available = ", ".join(sorted(models))
            raise ValueError(f"Unknown model version '{requested_version}'. Available: {available}")

        entry = models[requested_version]
        base_path = self.registry_path.parent
        return ModelInfo(
            version=requested_version,
            image_model_path=base_path / entry["image_model_path"],
            classifier_path=base_path / entry["classifier_path"],
            metadata_encoder_path=base_path / entry["metadata_encoder_path"],
            metadata_dim=int(entry["metadata_dim"]),
            metrics=entry.get("metrics", {}),
        )


class DigitClassifier:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self._cache: dict[str, tuple[CNNEncoder, FinalClassifier, object]] = {}

    @property
    def active_version(self) -> str:
        return self.registry.active_version

    def _load_model(self, version: str | None) -> tuple[str, CNNEncoder, FinalClassifier, object]:
        info = self.registry.resolve(version)
        if info.version not in self._cache:
            for path in (info.image_model_path, info.classifier_path, info.metadata_encoder_path):
                if not path.exists():
                    raise FileNotFoundError(f"Model artifact not found at {path}")

            image_model = CNNEncoder()
            image_model.load_state_dict(torch.load(info.image_model_path, map_location="cpu"))
            image_model.eval()

            final_model = FinalClassifier(metadata_dim=info.metadata_dim)
            final_model.load_state_dict(torch.load(info.classifier_path, map_location="cpu"))
            final_model.eval()

            metadata_encoder = joblib.load(info.metadata_encoder_path)
            self._cache[info.version] = (image_model, final_model, metadata_encoder)
        image_model, final_model, metadata_encoder = self._cache[info.version]
        return info.version, image_model, final_model, metadata_encoder

    def predict(
        self,
        image_array: np.ndarray,
        metadata: PredictionMetadata,
    ) -> tuple[int, float, dict[str, float], str]:
        model_version, image_model, final_model, metadata_encoder = self._load_model(
            metadata.model_version
        )
        img_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        meta_df = pd.DataFrame(
            [
                {
                    "pen_pressure": metadata.pen_pressure,
                    "writer_age": metadata.writer_age,
                    "handedness": metadata.handedness,
                }
            ]
        )
        meta_encoded = metadata_encoder.transform(meta_df).astype("float32")
        meta_tensor = torch.tensor(meta_encoded)

        with torch.no_grad():
            image_features = image_model(img_tensor)
            logits = final_model(image_features, meta_tensor)
            probabilities_tensor = torch.softmax(logits, dim=1)[0]

        probabilities = probabilities_tensor.detach().cpu().numpy()
        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])
        probability_map = {str(index): float(value) for index, value in enumerate(probabilities)}
        return prediction, confidence, probability_map, model_version
