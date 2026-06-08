from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter

from .data import ImageRecord, open_image


def extract_features(image: Image.Image, size: int = 64) -> np.ndarray:
    """Extract compact intensity, histogram, edge, and low-resolution image features."""
    gray = image.convert("L").resize((size, size))
    pixels = np.asarray(gray, dtype=np.float32) / 255.0
    histogram, _ = np.histogram(pixels, bins=16, range=(0.0, 1.0), density=True)
    edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    thumbnail = np.asarray(gray.resize((16, 16)), dtype=np.float32).reshape(-1) / 255.0
    statistics = np.array(
        [
            pixels.mean(),
            pixels.std(),
            np.quantile(pixels, 0.1),
            np.median(pixels),
            np.quantile(pixels, 0.9),
            edges.mean(),
            edges.std(),
        ],
        dtype=np.float32,
    )
    return np.concatenate([statistics, histogram.astype(np.float32), thumbnail])


def build_feature_matrix(
    records: Iterable[ImageRecord],
    noise_sigma: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features: list[np.ndarray] = []
    labels: list[int] = []
    for record in records:
        image = open_image(Path(record.path))
        if noise_sigma > 0:
            array = np.asarray(image, dtype=np.float32) / 255.0
            array = np.clip(array + rng.normal(0, noise_sigma, array.shape), 0, 1)
            image = Image.fromarray((array * 255).astype(np.uint8), mode="L")
        features.append(extract_features(image))
        labels.append(record.label)
    return np.stack(features), np.asarray(labels, dtype=np.int64)

