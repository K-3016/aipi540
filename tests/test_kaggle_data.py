from pathlib import Path

from PIL import Image

from brain_tumor_ml.kaggle_data import download_and_prepare_kaggle_dataset


def _write_source(root: Path) -> None:
    for split in ("Training", "Testing"):
        for class_name in ("glioma", "meningioma", "pituitary", "notumor"):
            directory = root / split / class_name
            directory.mkdir(parents=True)
            count = 4 if split == "Training" else 2
            for index in range(count):
                Image.new("L", (12, 12), color=index * 20).save(directory / f"{index}.jpg")


def test_prepare_kaggle_dataset_preserves_test_and_maps_notumor(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "prepared"
    _write_source(source)

    counts = download_and_prepare_kaggle_dataset(
        output_dir=output,
        validation_fraction=0.25,
        seed=7,
        source_dir=source,
    )

    assert counts == {"train": 12, "val": 4, "test": 8}
    assert (output / "train" / "normal").is_dir()
    assert len(list((output / "test" / "normal").glob("*.jpg"))) == 2
