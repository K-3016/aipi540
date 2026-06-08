from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .artifacts import load_deep, load_sklearn, save_json
from .data import ImageRecord, load_manifest
from .features import build_feature_matrix
from .metrics import classification_metrics
from .training import predict_deep


def add_noise_records(
    records: list[ImageRecord], output_dir: Path, sigma: float, seed: int
) -> list[ImageRecord]:
    if sigma == 0:
        return records
    rng = np.random.default_rng(seed)
    noisy_records: list[ImageRecord] = []
    sigma_dir = output_dir / f"noise_{sigma:.2f}"
    sigma_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        with Image.open(record.path) as source:
            array = np.asarray(source.convert("L"), dtype=np.float32) / 255.0
        noisy = np.clip(array + rng.normal(0, sigma, array.shape), 0, 1)
        path = sigma_dir / f"{index:05d}.png"
        Image.fromarray((noisy * 255).astype(np.uint8), mode="L").save(path)
        noisy_records.append(
            ImageRecord(
                path=str(path.resolve()),
                label=record.label,
                class_name=record.class_name,
                patient_id=record.patient_id,
                split=record.split,
            )
        )
    return noisy_records


def run_experiment(
    artifact_dir: Path,
    output_dir: Path,
    noise_levels: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20),
    seed: int = 42,
) -> list[dict]:
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_records = [
        record for record in load_manifest(artifact_dir / "manifest.csv") if record.split == "test"
    ]
    baseline = load_sklearn(artifact_dir / "baseline.joblib")
    classical = load_sklearn(artifact_dir / "classical.joblib")
    deep, image_size, _ = load_deep(artifact_dir / "deep_model.pt")
    rows: list[dict] = []

    for sigma in noise_levels:
        records = add_noise_records(test_records, output_dir / "noisy_images", sigma, seed)
        features, labels = build_feature_matrix(records)
        predictions = {
            "naive_baseline": baseline.predict_proba(features),
            "classical_logistic_regression": classical.predict_proba(features),
            "deep_cnn": predict_deep(deep, records, image_size=image_size),
        }
        for model_name, probabilities in predictions.items():
            metrics = classification_metrics(labels, probabilities)
            rows.append({"model": model_name, "noise_sigma": sigma, **metrics})

    save_json(rows, output_dir / "noise_robustness.json")
    _plot_results(rows, output_dir / "noise_robustness.png")
    return rows


def _plot_results(rows: list[dict], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for model_name in sorted({row["model"] for row in rows}):
        selected = [row for row in rows if row["model"] == model_name]
        axis.plot(
            [row["noise_sigma"] for row in selected],
            [row["balanced_accuracy"] for row in selected],
            marker="o",
            label=model_name,
        )
    axis.set(xlabel="Gaussian noise sigma", ylabel="Balanced accuracy", ylim=(0, 1.02))
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the image-noise robustness experiment.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/outputs"))
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0, 0.05, 0.1, 0.2])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = run_experiment(
        args.artifact_dir, args.output_dir, tuple(args.noise_levels), seed=args.seed
    )
    print(f"Wrote {len(rows)} experiment results to {args.output_dir}.")


if __name__ == "__main__":
    main()
