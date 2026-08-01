"""Clean MedExplain records, format prompts, and create reproducible splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.make_dataset import REQUIRED_FIELDS, load_jsonl, normalize_text, write_jsonl
except ModuleNotFoundError:  # Supports `python scripts/build_features.py`.
    from make_dataset import REQUIRED_FIELDS, load_jsonl, normalize_text, write_jsonl

LOGGER = logging.getLogger(__name__)
TASK_TEXT = "Rewrite the following medical statement in patient-friendly language."


def format_inference_prompt(medical_text: str) -> str:
    """Format a model prompt without exposing the reference answer."""
    text = normalize_text(medical_text)
    if not text:
        raise ValueError("medical_text must not be empty")
    return f"### Task\n{TASK_TEXT}\n\n### Medical statement\n{text}\n\n### Patient-friendly explanation\n"


def format_training_prompt(medical_text: str, output: str) -> str:
    """Format a complete supervised instruction-tuning example."""
    target = normalize_text(output)
    if not target:
        raise ValueError("output must not be empty")
    return f"{format_inference_prompt(medical_text)}{target}"


def canonical_key(record: dict[str, Any]) -> str:
    """Hash normalized source text to detect duplicates and split leakage."""
    normalized = re.sub(r"\W+", " ", normalize_text(record.get("input")).lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def prepare_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize fields, reject malformed records, and add prompt columns."""
    prepared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        item = dict(record)
        for field in REQUIRED_FIELDS:
            item[field] = normalize_text(item.get(field))
        if any(not item[field] for field in REQUIRED_FIELDS):
            continue
        key = canonical_key(item)
        if key in seen:
            continue
        seen.add(key)
        item["medical_text"] = item.pop("input")
        item["reference_output"] = item.pop("output")
        item["inference_prompt"] = format_inference_prompt(item["medical_text"])
        item["text"] = format_training_prompt(item["medical_text"], item["reference_output"])
        item["split_group"] = key
        prepared.append(item)
    return prepared


def split_records(
    records: list[dict[str, Any]], seed: int = 42
) -> dict[str, list[dict[str, Any]]]:
    """Create deterministic 80/10/10 splits with no source overlap."""
    if len(records) < 3:
        raise ValueError("At least 3 unique records are required to create three splits.")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    validation_size = max(1, round(total * 0.10))
    test_size = max(1, round(total * 0.10))
    if validation_size + test_size >= total:
        validation_size = test_size = 1
    train_size = total - validation_size - test_size
    return {
        "train": shuffled[:train_size],
        "validation": shuffled[train_size : train_size + validation_size],
        "test": shuffled[train_size + validation_size :],
    }


def build_features(input_path: Path, output_dir: Path, seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    """Run preprocessing and save split JSONL files."""
    records = prepare_records(load_jsonl(input_path))
    splits = split_records(records, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        write_jsonl(split, output_dir / f"{name}.jsonl")
        LOGGER.info("Saved %d %s examples", len(split), name)
    manifest = {
        "seed": seed,
        "counts": {name: len(split) for name, split in splits.items()},
        "source": str(input_path),
        "split_group": "SHA-256 of normalized medical input",
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return splits


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/medical_rewrite_dataset.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run feature construction."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    build_features(args.input, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
