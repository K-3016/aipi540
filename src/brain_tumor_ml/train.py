from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .artifacts import save_deep, save_json, save_sklearn
from .constants import CLASS_NAMES, DEFAULT_ARTIFACT_DIR, DEFAULT_IMAGE_SIZE
from .data import discover_images, save_manifest, split_records
from .error_analysis import save_error_analysis
from .features import build_feature_matrix
from .metrics import classification_metrics
from .training import (
    fit_baseline_features,
    fit_classical_features,
    fit_deep,
    predict_deep,
)


def run_training(
    data_dir: Path,
    output_dir: Path,
    patient_id_regex: str | None = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
    epochs: int = 15,
    batch_size: int = 16,
    seed: int = 42,
    architecture: str = "small_cnn",
    pretrained: bool = False,
    triage_confidence_threshold: float = 0.7,
    report_dir: Path = Path("data/outputs"),
) -> dict:
    if not 0 < triage_confidence_threshold < 1:
        raise ValueError("triage_confidence_threshold must be between 0 and 1.")
    output_dir = Path(output_dir)
    overall_started = time.perf_counter()
    print(f"Loading dataset from {data_dir}...", flush=True)
    records = split_records(
        discover_images(Path(data_dir), patient_id_regex=patient_id_regex),
        seed=seed,
    )
    save_manifest(records, output_dir / "manifest.csv")
    train_records = [record for record in records if record.split == "train"]
    val_records = [record for record in records if record.split == "val"]
    test_records = [record for record in records if record.split == "test"]
    print(
        f"Dataset ready: {len(train_records)} train, {len(val_records)} validation, "
        f"{len(test_records)} test images.",
        flush=True,
    )
    print("Extracting handcrafted train and test features (one-time pass)...", flush=True)
    x_train, y_train = build_feature_matrix(train_records)
    x_test, y_test = build_feature_matrix(test_records)

    results: dict[str, dict] = {}
    started = time.perf_counter()
    print("Training model 1/3: naive baseline...", flush=True)
    baseline = fit_baseline_features(x_train, y_train)
    save_sklearn(baseline, output_dir / "baseline.joblib")
    results["naive_baseline"] = classification_metrics(y_test, baseline.predict_proba(x_test))
    print(f"Naive baseline complete in {time.perf_counter() - started:.1f}s.", flush=True)

    started = time.perf_counter()
    print("Training model 2/3: logistic regression...", flush=True)
    classical = fit_classical_features(x_train, y_train, seed=seed)
    save_sklearn(classical, output_dir / "classical.joblib")
    results["classical_logistic_regression"] = classification_metrics(
        y_test, classical.predict_proba(x_test)
    )
    print(f"Logistic regression complete in {time.perf_counter() - started:.1f}s.", flush=True)

    started = time.perf_counter()
    print("Training model 3/3: CNN...", flush=True)
    deep_result = fit_deep(
        train_records,
        val_records,
        image_size=image_size,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        architecture=architecture,
        pretrained=pretrained,
    )
    save_deep(
        deep_result.model,
        output_dir / "deep_model.pt",
        image_size=image_size,
        architecture=architecture,
    )
    deep_probabilities = predict_deep(
        deep_result.model, test_records, image_size=image_size, batch_size=batch_size
    )
    results["deep_cnn"] = classification_metrics(y_test, deep_probabilities)
    print(f"CNN training and evaluation complete in {time.perf_counter() - started:.1f}s.", flush=True)
    save_error_analysis(test_records, deep_probabilities, Path(report_dir) / "error_analysis")
    save_json(deep_result.history, output_dir / "training_history.json")

    metadata = {
        "task": "multiclass_image_classification",
        "class_names": list(CLASS_NAMES),
        "deployed_model": "deep_cnn",
        "deep_architecture": architecture,
        "pretrained": pretrained,
        "triage_confidence_threshold": triage_confidence_threshold,
        "image_size": image_size,
        "seed": seed,
        "split_counts": {
            split: sum(record.split == split for record in records)
            for split in ("train", "val", "test")
        },
        "patient_grouping": bool(patient_id_regex),
        "patient_id_regex": patient_id_regex,
        "medical_use": "research_and_education_only",
    }
    save_json(metadata, output_dir / "metadata.json")
    save_json(results, output_dir / "metrics.json")
    _save_model_comparison(results, output_dir / "model_comparison.csv")
    _save_confusion_matrices(results, Path(report_dir) / "confusion_matrices.png")
    print(f"Training workflow complete in {time.perf_counter() - overall_started:.1f}s.", flush=True)
    return results


def _save_model_comparison(results: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
            ),
        )
        writer.writeheader()
        for model_name, metrics in results.items():
            writer.writerow(
                {
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["macro_precision"],
                    "recall": metrics["macro_recall"],
                    "f1_score": metrics["macro_f1"],
                    "roc_auc": metrics["roc_auc_ovr_macro"],
                }
            )


def _save_confusion_matrices(results: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(results), figsize=(12, 3.8))
    for axis, (model_name, metrics) in zip(axes, results.items(), strict=True):
        matrix = np.asarray(metrics["confusion_matrix"])
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set_title(model_name.replace("_", " ").title(), fontsize=10)
        axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
        axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    fig.colorbar(image, ax=axes, shrink=0.7)
    fig.subplots_adjust(left=0.07, right=0.94, bottom=0.25, top=0.86, wspace=0.5)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all required brain tumor classifiers.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--patient-id-regex")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--architecture", choices=("small_cnn", "resnet18"), default="small_cnn"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use ImageNet weights for ResNet18 (may download weights on first use).",
    )
    parser.add_argument(
        "--triage-confidence-threshold",
        type=float,
        default=0.7,
        help="Predictions below this confidence are flagged for human review.",
    )
    parser.add_argument("--report-dir", type=Path, default=Path("data/outputs"))
    args = parser.parse_args()
    results = run_training(**vars(args))
    for model_name, metrics in results.items():
        print(
            f"{model_name}: balanced_accuracy={metrics['balanced_accuracy']:.3f}, "
            f"macro_f1={metrics['macro_f1']:.3f}, "
            f"roc_auc_ovr_macro={metrics['roc_auc_ovr_macro']}"
        )


if __name__ == "__main__":
    main()
