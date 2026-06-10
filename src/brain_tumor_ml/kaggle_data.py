from __future__ import annotations

import random
import shutil
from pathlib import Path

from .constants import SUPPORTED_EXTENSIONS

DATASET_HANDLE = "masoudnickparvar/brain-tumor-mri-dataset"
CLASS_ALIASES = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "notumor": "normal",
    "no_tumor": "normal",
    "normal": "normal",
}


def download_and_prepare_kaggle_dataset(
    output_dir: Path = Path("data/processed/kaggle"),
    validation_fraction: float = 0.15,
    seed: int = 42,
    source_dir: Path | None = None,
) -> dict[str, int]:
    """Download the Kaggle dataset and create explicit train/val/test class folders."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if source_dir is None:
        try:
            import kagglehub
        except ImportError as error:
            raise RuntimeError(
                "kagglehub is required. Install dependencies with "
                "'pip install -r requirements.txt'."
            ) from error
        source_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))
    else:
        source_dir = Path(source_dir)

    training_dir = _find_named_directory(source_dir, "Training")
    testing_dir = _find_named_directory(source_dir, "Testing")
    output_dir = Path(output_dir)
    rng = random.Random(seed)
    counts = {split: 0 for split in ("train", "val", "test")}
    for split in counts:
        split_dir = output_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)

    for source_class_dir in _class_directories(training_dir):
        class_name = _canonical_class_name(source_class_dir.name)
        images = _images(source_class_dir)
        if len(images) < 2:
            raise ValueError(f"Class '{source_class_dir.name}' needs at least 2 training images.")
        rng.shuffle(images)
        validation_count = max(1, round(len(images) * validation_fraction))
        validation = set(images[:validation_count])
        for image in images:
            split = "val" if image in validation else "train"
            _copy_image(image, output_dir / split / class_name)
            counts[split] += 1

    for source_class_dir in _class_directories(testing_dir):
        class_name = _canonical_class_name(source_class_dir.name)
        for image in _images(source_class_dir):
            _copy_image(image, output_dir / "test" / class_name)
            counts["test"] += 1

    (output_dir / "DATASET_SOURCE.txt").write_text(
        f"Kaggle dataset: https://www.kaggle.com/datasets/{DATASET_HANDLE}\n"
        f"Downloaded source: {source_dir.resolve()}\n"
        f"Validation fraction from original Training folder: {validation_fraction}\n"
        f"Random seed: {seed}\n",
        encoding="utf-8",
    )
    return counts


def _find_named_directory(root: Path, name: str) -> Path:
    matches = [
        path
        for path in root.rglob("*")
        if path.is_dir() and path.name.casefold() == name.casefold()
    ]
    if not matches:
        raise FileNotFoundError(f"Could not find a '{name}' directory under '{root}'.")
    return min(matches, key=lambda path: len(path.parts))


def _class_directories(split_dir: Path) -> list[Path]:
    directories = [
        path
        for path in split_dir.iterdir()
        if path.is_dir() and path.name.casefold() in CLASS_ALIASES
    ]
    canonical = {_canonical_class_name(path.name) for path in directories}
    expected = {"glioma", "meningioma", "pituitary", "normal"}
    if canonical != expected:
        raise ValueError(
            f"Expected four classes under '{split_dir}'; found {sorted(canonical)}."
        )
    return sorted(directories)


def _canonical_class_name(name: str) -> str:
    try:
        return CLASS_ALIASES[name.casefold()]
    except KeyError as error:
        raise ValueError(f"Unsupported class folder '{name}'.") from error


def _images(directory: Path) -> list[Path]:
    return [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def _copy_image(source: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if destination.exists() and destination.resolve() != source.resolve():
        destination = destination_dir / f"{source.parent.name}_{source.name}"
    shutil.copy2(source, destination)
