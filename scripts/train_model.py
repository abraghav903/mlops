import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


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


def generate_metadata(n: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {
        "pen_pressure": rng.uniform(0.5, 1.5, n),
        "writer_age": rng.integers(10, 80, n),
        "handedness": rng.choice(["left", "right"], n),
    }


def save_example_files(img_array: np.ndarray, metadata_row: dict, output_dir: Path) -> None:
    image = Image.fromarray((img_array.squeeze() * 255).astype("uint8"), mode="L")
    image.save(output_dir / "example_digit.png")
    (output_dir / "example_metadata.json").write_text(
        json.dumps(metadata_row, indent=2),
        encoding="utf-8",
    )


def load_training_data(limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    transform = transforms.Compose([transforms.ToTensor()])
    try:
        dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
        images = dataset.data.numpy().astype(np.float32) / 255.0
        labels = dataset.targets.numpy()
    except RuntimeError:
        digits = load_digits()
        resized_images = []
        for image in digits.images:
            pil_image = Image.fromarray((image / 16.0 * 255).astype("uint8"), mode="L")
            resized = pil_image.resize((28, 28), Image.Resampling.LANCZOS)
            resized_images.append(np.asarray(resized, dtype=np.float32) / 255.0)
        images = np.asarray(resized_images)
        labels = digits.target

    if limit:
        images = images[:limit]
        labels = labels[:limit]

    return images, labels


def train(version: str, output_dir: Path, epochs: int, batch_size: int, limit: int | None) -> Path:
    images, labels = load_training_data(limit)
    images = np.expand_dims(images, 1)
    metadata_df = pd.DataFrame(generate_metadata(len(images)))

    encoder = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["pen_pressure", "writer_age"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["handedness"]),
        ]
    )
    meta_encoded = encoder.fit_transform(metadata_df).astype(np.float32)

    x_img = torch.tensor(images)
    x_meta = torch.tensor(meta_encoded)
    y = torch.tensor(labels, dtype=torch.long)

    dataset_tensor = TensorDataset(x_img, x_meta, y)
    loader = DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True)

    image_model = CNNEncoder()
    final_model = FinalClassifier(metadata_dim=x_meta.shape[1])
    optimizer = optim.Adam(
        list(image_model.parameters()) + list(final_model.parameters()),
        lr=1e-3,
    )
    loss_fn = nn.CrossEntropyLoss()

    image_model.train()
    final_model.train()
    for _ in range(epochs):
        for batch_images, batch_meta, batch_labels in loader:
            optimizer.zero_grad()
            image_features = image_model(batch_images)
            logits = final_model(image_features, batch_meta)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()

    image_model.eval()
    final_model.eval()
    with torch.no_grad():
        logits = final_model(image_model(x_img), x_meta)
        predictions = torch.argmax(logits, dim=1).numpy()
    accuracy = float(accuracy_score(labels, predictions))

    version_dir = output_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    torch.save(image_model.state_dict(), version_dir / "image_model.pth")
    torch.save(final_model.state_dict(), version_dir / "final_classifier.pth")
    joblib.dump(encoder, version_dir / "metadata_encoder.joblib")
    save_example_files(images[0], metadata_df.iloc[0].to_dict(), version_dir)

    registry_path = output_dir / "registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {"active_version": version, "models": {}}

    registry["active_version"] = version
    registry.setdefault("models", {})[version] = {
        "image_model_path": f"{version}/image_model.pth",
        "classifier_path": f"{version}/final_classifier.pth",
        "metadata_encoder_path": f"{version}/metadata_encoder.joblib",
        "metadata_dim": int(x_meta.shape[1]),
        "created_at": datetime.now(UTC).isoformat(),
        "metrics": {"training_accuracy": accuracy},
    }
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register the notebook digit model.")
    parser.add_argument("--version", default="v1", help="Model version to write to the registry.")
    parser.add_argument(
        "--output-dir",
        default="models",
        type=Path,
        help="Model registry directory.",
    )
    parser.add_argument("--epochs", default=1, type=int, help="Training epochs.")
    parser.add_argument("--batch-size", default=128, type=int, help="Training batch size.")
    parser.add_argument(
        "--limit",
        default=10_000,
        type=int,
        help="Limit training rows for quick local builds. Use 0 for full MNIST.",
    )
    args = parser.parse_args()

    limit = args.limit or None
    registry_path = train(args.version, args.output_dir, args.epochs, args.batch_size, limit)
    print(f"Wrote model registry to {registry_path}")


if __name__ == "__main__":
    main()
