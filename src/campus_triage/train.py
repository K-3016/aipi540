# Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""Training entry points for all Campus Triage models.

Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from campus_triage.config import BASELINE_MODEL_PATH, CLASSICAL_MODEL_PATH, MODELS_DIR, TRANSFORMER_MODEL_DIR
from campus_triage.data import create_and_save_dataset, load_processed_splits
from campus_triage.models import (
    build_baseline_model,
    build_classical_model,
    save_dual_classifier,
    transformer_dependencies_available,
)


def train_baseline(train_dataframe: pd.DataFrame) -> Path:
    """Train and save the majority baseline."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = build_baseline_model(train_dataframe)
    save_dual_classifier(model, str(BASELINE_MODEL_PATH))
    return BASELINE_MODEL_PATH


def train_classical(train_dataframe: pd.DataFrame) -> Path:
    """Train and save the TF-IDF Logistic Regression model."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = build_classical_model(train_dataframe)
    save_dual_classifier(model, str(CLASSICAL_MODEL_PATH))
    return CLASSICAL_MODEL_PATH


def train_transformer_optional(train_dataframe: pd.DataFrame, validation_dataframe: pd.DataFrame, max_rows: int = 600) -> str:
    """Fine-tune lightweight DistilBERT models when optional dependencies are available."""

    if not transformer_dependencies_available():
        return "Skipped transformer training because optional dependencies are not installed."

    from campus_triage.transformer_training import train_transformer_models

    TRANSFORMER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_transformer_models(train_dataframe, validation_dataframe, TRANSFORMER_MODEL_DIR, max_rows=max_rows)
    return f"Saved transformer models to {TRANSFORMER_MODEL_DIR}"


def train_all(include_transformer: bool = False) -> list[str]:
    """Train all required models, with transformer training optional for laptop speed."""

    if not Path("data/processed/train.csv").exists():
        create_and_save_dataset()
    train_dataframe, validation_dataframe, _ = load_processed_splits()
    results = [
        f"Saved baseline model to {train_baseline(train_dataframe)}",
        f"Saved classical model to {train_classical(train_dataframe)}",
    ]
    if include_transformer:
        results.append(train_transformer_optional(train_dataframe, validation_dataframe))
    else:
        results.append("Skipped transformer training by default. Use --include-transformer to run it.")
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Train Campus Triage models.")
    parser.add_argument("--include-transformer", action="store_true", help="Fine-tune optional DistilBERT models.")
    return parser.parse_args()


def main() -> None:
    """Run training from the command line."""

    args = parse_args()
    for message in train_all(include_transformer=args.include_transformer):
        print(message)


if __name__ == "__main__":
    main()
