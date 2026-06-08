from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import save_deep, save_json, save_sklearn
from .constants import CLASS_NAMES, DEFAULT_ARTIFACT_DIR, DEFAULT_IMAGE_SIZE
from .data import discover_images, save_manifest, split_records
from .features import build_feature_matrix
from .metrics import classification_metrics
from .training import fit_baseline, fit_classical, fit_deep, predict_deep


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
) -> dict:
    if not 0 < triage_confidence_threshold < 1:
        raise ValueError("triage_confidence_threshold must be between 0 and 1.")
    output_dir = Path(output_dir)
    records = split_records(
        discover_images(Path(data_dir), patient_id_regex=patient_id_regex),
        seed=seed,
    )
    save_manifest(records, output_dir / "manifest.csv")
    train_records = [record for record in records if record.split == "train"]
    val_records = [record for record in records if record.split == "val"]
    test_records = [record for record in records if record.split == "test"]
    x_test, y_test = build_feature_matrix(test_records)

    results: dict[str, dict] = {}
    baseline = fit_baseline(train_records)
    save_sklearn(baseline, output_dir / "baseline.joblib")
    results["naive_baseline"] = classification_metrics(y_test, baseline.predict_proba(x_test))

    classical = fit_classical(train_records, seed=seed)
    save_sklearn(classical, output_dir / "classical.joblib")
    results["classical_logistic_regression"] = classification_metrics(
        y_test, classical.predict_proba(x_test)
    )

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
    return results


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
