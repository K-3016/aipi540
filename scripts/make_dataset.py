"""Prepare dataset directories or generate synthetic smoke-test data."""

from __future__ import annotations

import argparse
from pathlib import Path

from brain_tumor_ml.constants import CLASS_NAMES
from brain_tumor_ml.demo_data import generate_demo_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--demo-dir", type=Path, default=Path("data/processed/demo"))
    parser.add_argument("--samples-per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for class_name in CLASS_NAMES:
        (args.raw_dir / class_name).mkdir(parents=True, exist_ok=True)
    if args.demo:
        generate_demo_data(args.demo_dir, args.samples_per_class, args.seed)
        print(f"Generated synthetic smoke-test images in {args.demo_dir}.")
    print(f"Raw dataset folders are ready in {args.raw_dir}.")


if __name__ == "__main__":
    main()
