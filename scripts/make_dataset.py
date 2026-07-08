# Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""Generate synthetic campus support data.

Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from campus_triage.data import create_and_save_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Generate the synthetic campus support dataset.")
    parser.add_argument("--rows", type=int, default=1800, help="Number of synthetic rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    """Generate and save dataset files."""

    args = parse_args()
    dataframe = create_and_save_dataset(row_count=args.rows, seed=args.seed)
    print(f"Generated {len(dataframe)} rows and saved raw/processed CSV files.")


if __name__ == "__main__":
    main()
