from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from .artifacts import load_deep, load_json
from .data import build_transform
from .explainability import grad_cam, overlay_heatmap


@dataclass
class PredictionService:
    artifact_dir: Path

    def __post_init__(self) -> None:
        self.artifact_dir = Path(self.artifact_dir)
        self.metadata = load_json(self.artifact_dir / "metadata.json")
        self.model, self.image_size, self.architecture = load_deep(
            self.artifact_dir / "deep_model.pt"
        )
        self.transform = build_transform(self.image_size)

    def predict(self, image: Image.Image) -> dict:
        tensor = self.transform(image.convert("L")).unsqueeze(0)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1).squeeze(0).tolist()
        predicted_index = int(torch.tensor(probabilities).argmax().item())
        confidence = float(probabilities[predicted_index])
        threshold = float(self.metadata.get("triage_confidence_threshold", 0.7))
        return {
            "predicted_class": self.metadata["class_names"][predicted_index],
            "confidence": confidence,
            "probabilities": {
                class_name: float(probability)
                for class_name, probability in zip(
                    self.metadata["class_names"], probabilities, strict=True
                )
            },
            "review_recommended": confidence < threshold,
            "triage_confidence_threshold": threshold,
            "model_architecture": self.architecture,
            "disclaimer": (
                "Research-use output only. This model is not a diagnostic device and must not "
                "replace review by qualified clinicians."
            ),
        }

    def explain(self, image: Image.Image) -> tuple[dict, Image.Image]:
        prediction = self.predict(image)
        class_index = self.metadata["class_names"].index(prediction["predicted_class"])
        tensor = self.transform(image.convert("L")).unsqueeze(0)
        heatmap = grad_cam(self.model, tensor, class_index=class_index)
        return prediction, overlay_heatmap(image, heatmap)
