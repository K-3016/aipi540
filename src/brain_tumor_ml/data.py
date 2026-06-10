from __future__ import annotations

import csv
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .constants import CLASS_NAMES, CLASS_TO_INDEX, DEFAULT_IMAGE_SIZE, SUPPORTED_EXTENSIONS


@dataclass(frozen=True)
class ImageRecord:
    path: str
    label: int
    class_name: str
    patient_id: str
    split: str = ""


def discover_images(data_dir: Path, patient_id_regex: str | None = None) -> list[ImageRecord]:
    """Discover class-folder images and optionally extract patient IDs from filenames."""
    data_dir = Path(data_dir)
    pattern = re.compile(patient_id_regex) if patient_id_regex else None
    records: list[ImageRecord] = []

    split_directories = [
        (split, data_dir / split)
        for split in ("train", "val", "test")
        if (data_dir / split).is_dir()
    ]
    roots = split_directories or [("", data_dir)]
    for split, root in roots:
        for class_name in CLASS_NAMES:
            class_dir = root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected class directory '{class_dir}'. "
                    f"Required folders are: {', '.join(CLASS_NAMES)}."
                )
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    patient_id = path.stem
                    if pattern:
                        match = pattern.search(path.name)
                        if not match:
                            raise ValueError(f"Patient ID regex did not match '{path.name}'.")
                        patient_id = match.group(1) if match.groups() else match.group(0)
                    records.append(
                        ImageRecord(
                            path=str(path.resolve()),
                            label=CLASS_TO_INDEX[class_name],
                            class_name=class_name,
                            patient_id=patient_id,
                            split=split,
                        )
                    )

    if not records:
        raise ValueError(f"No supported images found under '{data_dir}'.")
    present = {record.class_name for record in records}
    if present != set(CLASS_NAMES):
        raise ValueError(f"All classes are required; found {sorted(present)}.")
    return records


def split_records(
    records: Iterable[ImageRecord],
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> list[ImageRecord]:
    """Create deterministic, class-stratified splits while keeping patient IDs together."""
    records = list(records)
    assigned = {record.split for record in records}
    if assigned == {"train", "val", "test"}:
        return sorted(records, key=lambda record: (record.split, record.path))
    if assigned != {""}:
        raise ValueError("Records must either all have explicit splits or all be unsplit.")
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be positive and sum to less than 1.")

    by_class_patient: dict[int, dict[str, list[ImageRecord]]] = {}
    for record in records:
        by_class_patient.setdefault(record.label, {}).setdefault(record.patient_id, []).append(record)

    rng = random.Random(seed)
    result: list[ImageRecord] = []
    for label, patient_records in sorted(by_class_patient.items()):
        patients = sorted(patient_records)
        if len(patients) < 3:
            raise ValueError(
                f"Class {CLASS_NAMES[label]} needs at least 3 unique patient IDs for train/val/test."
            )
        rng.shuffle(patients)
        n = len(patients)
        n_test = max(1, round(n * test_fraction))
        n_val = max(1, round(n * val_fraction))
        if n_test + n_val >= n:
            n_test, n_val = 1, 1

        split_by_patient = {
            patient: (
                "test"
                if index < n_test
                else "val"
                if index < n_test + n_val
                else "train"
            )
            for index, patient in enumerate(patients)
        }
        for patient, patient_group in patient_records.items():
            split = split_by_patient[patient]
            result.extend(ImageRecord(**{**asdict(record), "split": split}) for record in patient_group)
    return sorted(result, key=lambda record: (record.split, record.path))


def save_manifest(records: Iterable[ImageRecord], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ImageRecord.__dataclass_fields__.keys())
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def load_manifest(path: Path) -> list[ImageRecord]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        return [
            ImageRecord(
                path=row["path"],
                label=int(row["label"]),
                class_name=row["class_name"],
                patient_id=row["patient_id"],
                split=row["split"],
            )
            for row in rows
        ]


def build_transform(image_size: int = DEFAULT_IMAGE_SIZE, training: bool = False):
    operations: list[object] = [transforms.Grayscale(num_output_channels=1)]
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(8),
                transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),
            ]
        )
    operations.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return transforms.Compose(operations)


def open_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L").copy()


class BrainTumorDataset(Dataset):
    def __init__(
        self,
        records: Iterable[ImageRecord],
        image_size: int = DEFAULT_IMAGE_SIZE,
        training: bool = False,
    ) -> None:
        self.records = list(records)
        self.transform = build_transform(image_size=image_size, training=training)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        return self.transform(open_image(record.path)), np.int64(record.label)
