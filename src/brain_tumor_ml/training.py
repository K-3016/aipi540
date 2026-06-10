from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from .constants import CLASS_NAMES
from .data import BrainTumorDataset, ImageRecord
from .features import build_feature_matrix
from .models import build_model


@dataclass
class DeepTrainingResult:
    model: nn.Module
    history: list[dict[str, float]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_baseline(records: list[ImageRecord]) -> DummyClassifier:
    features, labels = build_feature_matrix(records)
    return fit_baseline_features(features, labels)


def fit_baseline_features(features: np.ndarray, labels: np.ndarray) -> DummyClassifier:
    model = DummyClassifier(strategy="prior")
    model.fit(features, labels)
    return model


def fit_classical(records: list[ImageRecord], seed: int = 42) -> Pipeline:
    features, labels = build_feature_matrix(records)
    return fit_classical_features(features, labels, seed=seed)


def fit_classical_features(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                    C=1.0,
                ),
            ),
        ]
    )
    model.fit(features, labels)
    return model


def fit_deep(
    train_records: list[ImageRecord],
    val_records: list[ImageRecord],
    image_size: int = 128,
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    patience: int = 4,
    seed: int = 42,
    architecture: str = "small_cnn",
    pretrained: bool = False,
    use_augmentation: bool = True,
) -> DeepTrainingResult:
    set_seed(seed)
    device = _device()
    model = build_model(architecture=architecture, pretrained=pretrained).to(device)
    print(
        f"CNN training on {device}: {len(train_records)} train, "
        f"{len(val_records)} validation images, up to {epochs} epochs.",
        flush=True,
    )
    train_loader = DataLoader(
        BrainTumorDataset(
            train_records,
            image_size=image_size,
            training=use_augmentation,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        BrainTumorDataset(val_records, image_size=image_size),
        batch_size=batch_size,
        shuffle=False,
    )
    labels = np.asarray([record.label for record in train_records], dtype=np.int64)
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    class_weights = len(labels) / (len(counts) * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = _run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"  epoch {epoch:02d}/{epochs}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"  early stopping after epoch {epoch}.", flush=True)
                break

    model.load_state_dict(best_state)
    return DeepTrainingResult(model=model.cpu(), history=history)


def predict_deep(
    model: nn.Module,
    records: list[ImageRecord],
    image_size: int = 128,
    batch_size: int = 32,
) -> np.ndarray:
    loader = DataLoader(
        BrainTumorDataset(records, image_size=image_size),
        batch_size=batch_size,
        shuffle=False,
    )
    device = _device()
    model = model.to(device).eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for images, _ in loader:
            probabilities.append(torch.softmax(model(images.to(device)), dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def _run_epoch(model, loader, criterion, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device, dtype=torch.long)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        total_examples += len(labels)
    return total_loss / max(1, total_examples)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
