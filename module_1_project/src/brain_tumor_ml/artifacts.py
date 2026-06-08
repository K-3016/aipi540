from __future__ import annotations

import json
from pathlib import Path

import joblib
import torch

from .constants import CLASS_NAMES
from .models import build_model


def save_json(payload: dict | list, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_sklearn(model, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_sklearn(path: Path):
    return joblib.load(path)


def save_deep(model, path: Path, image_size: int, architecture: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "image_size": image_size,
            "architecture": architecture,
            "num_classes": len(CLASS_NAMES),
        },
        path,
    )


def load_deep(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture", "small_cnn")
    model = build_model(
        architecture=architecture,
        num_classes=int(checkpoint.get("num_classes", len(CLASS_NAMES))),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, int(checkpoint["image_size"]), architecture
