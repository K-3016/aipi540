from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .artifacts import load_json, save_json
from .data import load_manifest
from .metrics import classification_metrics
from .training import fit_deep, predict_deep


def run_experiment(
    artifact_dir: Path,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 16,
    seed: int = 42,
) -> list[dict]:
    """Compare the same CNN trained with and without image augmentation."""
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_json(artifact_dir / "metadata.json")
    records = load_manifest(artifact_dir / "manifest.csv")
    train_records = [record for record in records if record.split == "train"]
    val_records = [record for record in records if record.split == "val"]
    test_records = [record for record in records if record.split == "test"]
    labels = np.asarray([record.label for record in test_records], dtype=np.int64)
    rows = []

    for use_augmentation in (False, True):
        condition = "with augmentation" if use_augmentation else "without augmentation"
        print(
            f"Experiment: training CNN {condition} for up to {epochs} epochs...",
            flush=True,
        )
        result = fit_deep(
            train_records,
            val_records,
            image_size=int(metadata["image_size"]),
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            architecture="small_cnn",
            pretrained=False,
            use_augmentation=use_augmentation,
        )
        probabilities = predict_deep(
            result.model,
            test_records,
            image_size=int(metadata["image_size"]),
            batch_size=batch_size,
        )
        metrics = classification_metrics(labels, probabilities)
        rows.append(
            {
                "augmentation": "with_augmentation" if use_augmentation else "without_augmentation",
                "accuracy": metrics["accuracy"],
                "precision": metrics["macro_precision"],
                "recall": metrics["macro_recall"],
                "f1_score": metrics["macro_f1"],
            }
        )
        print(
            f"Experiment result ({condition}): accuracy={metrics['accuracy']:.3f}, "
            f"macro_f1={metrics['macro_f1']:.3f}",
            flush=True,
        )

    save_json(
        {
            "question": "Does training-time image augmentation improve CNN performance?",
            "change": (
                "Random horizontal flip, rotation, and small translation were enabled while "
                "the architecture, split, seed, and training budget stayed fixed."
            ),
            "results": rows,
            "interpretation": _interpret(rows),
        },
        output_dir / "augmentation_experiment.json",
    )
    _save_csv(rows, output_dir / "augmentation_experiment.csv")
    _plot(rows, output_dir / "augmentation_experiment.png")
    return rows


def _interpret(rows: list[dict]) -> str:
    without = rows[0]["f1_score"]
    with_augmentation = rows[1]["f1_score"]
    difference = with_augmentation - without
    if difference > 0:
        return f"Augmentation improved macro F1 by {difference:.3f} on the test set."
    if difference < 0:
        return f"Augmentation reduced macro F1 by {abs(difference):.3f} on the test set."
    return "Augmentation produced no change in macro F1 on the test set."


def _save_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict], path: Path) -> None:
    labels = ["Without augmentation", "With augmentation"]
    metrics = ("accuracy", "recall", "f1_score")
    x = np.arange(len(labels))
    width = 0.24
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for index, metric in enumerate(metrics):
        axis.bar(
            x + (index - 1) * width,
            [row[metric] for row in rows],
            width,
            label=metric.replace("_", " ").title(),
        )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("CNN Performance With and Without Augmentation")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CNN training with and without augmentation.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/outputs"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = run_experiment(
        args.artifact_dir,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"Wrote {len(rows)} augmentation experiment results to {args.output_dir}.")


if __name__ == "__main__":
    main()
