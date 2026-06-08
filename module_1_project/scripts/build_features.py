"""Build and save classical-ML features from a class-folder dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from brain_tumor_ml.data import discover_images, save_manifest, split_records
from brain_tumor_ml.features import build_feature_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--patient-id-regex")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = split_records(
        discover_images(args.data_dir, patient_id_regex=args.patient_id_regex),
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(records, args.output_dir / "manifest.csv")
    for split in ("train", "val", "test"):
        selected = [record for record in records if record.split == split]
        features, labels = build_feature_matrix(selected)
        np.savez_compressed(
            args.output_dir / f"{split}_features.npz",
            features=features,
            labels=labels,
        )
    print(f"Saved processed features and split manifest in {args.output_dir}.")


if __name__ == "__main__":
    main()
