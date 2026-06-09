from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .constants import CLASS_NAMES
from .data import ImageRecord, open_image


def save_error_analysis(
    records: list[ImageRecord],
    probabilities: np.ndarray,
    output_dir: Path,
    limit: int = 5,
) -> list[dict]:
    """Save a small, reproducible report of the CNN's most confident mistakes."""
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    predictions = probabilities.argmax(axis=1)
    errors = [
        (index, float(probabilities[index, predictions[index]]))
        for index, record in enumerate(records)
        if predictions[index] != record.label
    ]
    errors.sort(key=lambda item: item[1], reverse=True)

    rows = []
    for rank, (index, confidence) in enumerate(errors[:limit], start=1):
        record = records[index]
        predicted_label = int(predictions[index])
        image = open_image(record.path)
        reason, improvement = _possible_reason(image, confidence)
        filename = f"error_{rank}_{record.class_name}_as_{CLASS_NAMES[predicted_label]}.png"
        _save_annotated_image(
            image,
            image_dir / filename,
            true_label=record.class_name,
            predicted_label=CLASS_NAMES[predicted_label],
            confidence=confidence,
        )
        rows.append(
            {
                "rank": rank,
                "source_image": record.path,
                "saved_image": str((image_dir / filename).resolve()),
                "true_label": record.class_name,
                "predicted_label": CLASS_NAMES[predicted_label],
                "confidence": confidence,
                "possible_reason": reason,
                "suggested_improvement": improvement,
            }
        )

    with (output_dir / "error_analysis.csv").open("w", newline="", encoding="utf-8") as file:
        fields = (
            "rank",
            "source_image",
            "saved_image",
            "true_label",
            "predicted_label",
            "confidence",
            "possible_reason",
            "suggested_improvement",
        )
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _possible_reason(image: Image.Image, confidence: float) -> tuple[str, str]:
    pixels = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    contrast = float(pixels.std())
    edges = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    sharpness = float(edges.std() / 255.0)
    if contrast < 0.12:
        return (
            "Low image contrast may hide tumor boundaries.",
            "Apply contrast normalization and include more low-contrast examples.",
        )
    if sharpness < 0.10:
        return (
            "The image may be blurry or have weak anatomical edges.",
            "Add image-quality checks and blur augmentation during training.",
        )
    if confidence < 0.55:
        return (
            "The model was uncertain, suggesting visual overlap between classes.",
            "Collect more examples for the confused classes and review their labels.",
        )
    return (
        "The classes may share similar shape or texture patterns in this slice.",
        "Use more training data, additional MRI slices, or a stronger feature extractor.",
    )


def _save_annotated_image(
    image: Image.Image,
    path: Path,
    true_label: str,
    predicted_label: str,
    confidence: float,
) -> None:
    canvas = image.convert("RGB").resize((320, 320))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 260, 320, 320), fill=(14, 24, 39))
    draw.text((10, 270), f"True: {true_label}", fill=(255, 255, 255))
    draw.text(
        (10, 292),
        f"Predicted: {predicted_label} ({confidence:.1%})",
        fill=(255, 190, 120),
    )
    canvas.save(path)
