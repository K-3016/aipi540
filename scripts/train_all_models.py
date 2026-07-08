# Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""Train all default Campus Triage models.

Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from campus_triage.train import train_all


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Train Campus Triage models.")
    parser.add_argument("--include-transformer", action="store_true", help="Also fine-tune optional DistilBERT models.")
    return parser.parse_args()


def main() -> None:
    """Train models and print artifact paths."""

    args = parse_args()
    for message in train_all(include_transformer=args.include_transformer):
        print(message)


if __name__ == "__main__":
    main()
