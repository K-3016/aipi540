"""Lightweight tests for dataset creation and split integrity."""

from __future__ import annotations

from scripts.build_features import prepare_records, split_records
from scripts.make_dataset import (
    INSTRUCTION,
    clean_records,
    create_synthetic_records,
    deduplicate_records,
    validate_record,
)


def _record(text: str, output: str = "Plain explanation.") -> dict[str, str]:
    return {"instruction": INSTRUCTION, "input": text, "output": output}


def test_required_fields_exist_in_synthetic_records() -> None:
    records = create_synthetic_records(25, seed=42)
    assert records
    assert all({"instruction", "input", "output", "example_id"} <= record.keys() for record in records)


def test_empty_records_are_rejected() -> None:
    valid, errors = validate_record(_record("", "Explanation"))
    assert not valid
    assert "missing_or_empty_input" in errors
    cleaned, rejected = clean_records([_record("", "Explanation")])
    assert cleaned == []
    assert rejected["missing_or_empty_input"] == 1


def test_duplicate_handling() -> None:
    duplicate = _record("The same source.")
    assert len(deduplicate_records([duplicate, dict(duplicate)])) == 1


def test_splits_do_not_overlap() -> None:
    records = prepare_records([_record(f"Medical finding {index}.", f"Explanation {index}.") for index in range(20)])
    splits = split_records(records, seed=42)
    groups = {name: {row["split_group"] for row in rows} for name, rows in splits.items()}
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])


def test_splits_are_reproducible() -> None:
    records = prepare_records([_record(f"Finding {index}.", f"Explanation {index}.") for index in range(20)])
    first = split_records(records, seed=7)
    second = split_records(records, seed=7)
    assert {
        name: [row["medical_text"] for row in rows] for name, rows in first.items()
    } == {
        name: [row["medical_text"] for row in rows] for name, rows in second.items()
    }

