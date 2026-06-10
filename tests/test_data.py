from pathlib import Path

from PIL import Image

from brain_tumor_ml.data import discover_images, split_records


def _write_dataset(root: Path) -> None:
    for class_name in ("glioma", "meningioma", "pituitary", "normal"):
        directory = root / class_name
        directory.mkdir(parents=True)
        for patient in range(6):
            for image_index in range(2):
                Image.new("L", (16, 16), color=patient * 20).save(
                    directory / f"patient_{patient:02d}_slice_{image_index}.png"
                )


def test_patient_groups_do_not_cross_splits(tmp_path):
    _write_dataset(tmp_path)
    records = discover_images(tmp_path, patient_id_regex=r"(patient_\d+)")
    split = split_records(records, seed=7, val_fraction=0.2, test_fraction=0.2)

    seen = {}
    for record in split:
        key = (record.class_name, record.patient_id)
        seen.setdefault(key, record.split)
        assert seen[key] == record.split
    assert {record.split for record in split} == {"train", "val", "test"}


def test_discovery_requires_expected_folders(tmp_path):
    (tmp_path / "glioma").mkdir()
    try:
        discover_images(tmp_path)
    except FileNotFoundError as error:
        assert "meningioma" in str(error)
    else:
        raise AssertionError("Expected missing class folder to fail.")


def test_explicit_split_directories_are_preserved(tmp_path):
    for split in ("train", "val", "test"):
        for class_name in ("glioma", "meningioma", "pituitary", "normal"):
            directory = tmp_path / split / class_name
            directory.mkdir(parents=True)
            Image.new("L", (16, 16), color=100).save(directory / f"{split}_{class_name}.png")

    records = discover_images(tmp_path)
    split = split_records(records)

    assert len(split) == 12
    assert {record.split for record in split} == {"train", "val", "test"}
