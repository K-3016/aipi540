# Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""Evaluation and error analysis for Campus Triage models.

Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from campus_triage.config import (
    BASELINE_MODEL_PATH,
    CATEGORY_LABELS,
    CLASSICAL_MODEL_PATH,
    OUTPUTS_DIR,
    TEST_PATH,
    TRAIN_PATH,
    URGENCY_LABELS,
)
from campus_triage.data import create_and_save_dataset
from campus_triage.models import (
    DualClassifier,
    load_dual_classifier,
    load_transformer_dual_classifier,
    transformer_model_available,
)
from campus_triage.train import train_all


def ensure_models_and_data() -> None:
    """Create data and default non-transformer models when they do not exist yet."""

    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        create_and_save_dataset()
    if not BASELINE_MODEL_PATH.exists() or not CLASSICAL_MODEL_PATH.exists():
        train_all(include_transformer=False)


def load_evaluation_models() -> list[DualClassifier]:
    """Load every trained model available for evaluation."""

    models = [
        load_dual_classifier(str(BASELINE_MODEL_PATH)),
        load_dual_classifier(str(CLASSICAL_MODEL_PATH)),
    ]
    if transformer_model_available():
        models.append(load_transformer_dual_classifier())
    return models


def label_metrics(y_true: pd.Series, y_pred: pd.Series, prefix: str) -> dict[str, float]:
    """Compute high-level metrics for one label task."""

    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}_weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def evaluate_dual_classifier(model: DualClassifier, test_dataframe: pd.DataFrame) -> dict[str, float | str]:
    """Evaluate category and urgency predictions for a model."""

    category_predictions, urgency_predictions = model.predict(test_dataframe)
    results: dict[str, float | str] = {"model": model.model_name}
    results.update(label_metrics(test_dataframe["category"], category_predictions, "category"))
    results.update(label_metrics(test_dataframe["urgency"], urgency_predictions, "urgency"))
    return results


def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, labels: list[str], title: str, output_path: Path) -> None:
    """Save a confusion matrix plot."""

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(9, 6))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    plt.title(title)
    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=35, ha="right")
    plt.yticks(tick_marks, labels)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            color = "white" if value > threshold else "black"
            plt.text(column_index, row_index, str(value), ha="center", va="center", color=color)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def build_reports(models: list[DualClassifier], test_dataframe: pd.DataFrame) -> str:
    """Build plain-text per-class classification reports."""

    report_sections = []
    for model in models:
        category_predictions, urgency_predictions = model.predict(test_dataframe)
        report_sections.append(f"Model: {model.model_name}\n")
        report_sections.append("Category report\n")
        report_sections.append(
            classification_report(test_dataframe["category"], category_predictions, labels=CATEGORY_LABELS, zero_division=0)
        )
        report_sections.append("\nUrgency report\n")
        report_sections.append(
            classification_report(test_dataframe["urgency"], urgency_predictions, labels=URGENCY_LABELS, zero_division=0)
        )
        report_sections.append("\n" + "=" * 80 + "\n")
    return "\n".join(report_sections)


def create_error_analysis(best_model: DualClassifier, test_dataframe: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Save five concrete mispredictions with root-cause and mitigation notes."""

    category_predictions, urgency_predictions = best_model.predict(test_dataframe)
    analysis_dataframe = test_dataframe.copy()
    analysis_dataframe["predicted_category"] = category_predictions
    analysis_dataframe["predicted_urgency"] = urgency_predictions
    mistakes = analysis_dataframe[
        (analysis_dataframe["category"] != analysis_dataframe["predicted_category"])
        | (analysis_dataframe["urgency"] != analysis_dataframe["predicted_urgency"])
    ].head(5)
    rows = []
    for _, row in mistakes.iterrows():
        root_cause = infer_root_cause(row["message_text"], row["category"], row["predicted_category"])
        rows.append(
            {
                "message_text": row["message_text"],
                "true_category": row["category"],
                "predicted_category": row["predicted_category"],
                "true_urgency": row["urgency"],
                "predicted_urgency": row["predicted_urgency"],
                "likely_root_cause": root_cause,
                "concrete_mitigation_strategy": mitigation_for_root_cause(root_cause),
            }
        )
    error_dataframe = pd.DataFrame(rows)
    error_dataframe.to_csv(output_path, index=False)
    return error_dataframe


def infer_root_cause(message_text: str, true_category: str, predicted_category: str) -> str:
    """Infer a simple root cause for a misprediction."""

    text = str(message_text)
    if len(text.split()) <= 4:
        return "Very short message lacks enough context for reliable routing."
    if true_category != predicted_category and any(word in text.lower() for word in ["help", "question", "portal"]):
        return "Ambiguous shared vocabulary appears in multiple support categories."
    if any(symbol in text for symbol in ["???", "!!", ":/"]):
        return "Informal punctuation and noisy phrasing may distort text features."
    return "The message contains limited category-specific keywords."


def mitigation_for_root_cause(root_cause: str) -> str:
    """Map a root cause to an actionable mitigation."""

    if "short" in root_cause:
        return "Ask for one clarifying detail before auto-routing short requests."
    if "Ambiguous" in root_cause:
        return "Add a human-review threshold for low-confidence cross-category cases."
    if "punctuation" in root_cause:
        return "Expand training augmentation with noisy chat-style messages."
    return "Collect more labeled examples with advisor-reviewed category keywords."


def run_evaluation() -> pd.DataFrame:
    """Evaluate all trained strategies and save required outputs."""

    ensure_models_and_data()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    test_dataframe = pd.read_csv(TEST_PATH)
    models = load_evaluation_models()
    comparison = pd.DataFrame([evaluate_dual_classifier(model, test_dataframe) for model in models])
    comparison.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)

    best_model_name = comparison.sort_values(["category_macro_f1", "urgency_macro_f1"], ascending=False).iloc[0]["model"]
    best_model = next(model for model in models if model.model_name == best_model_name)
    category_predictions, urgency_predictions = best_model.predict(test_dataframe)
    save_confusion_matrix(
        test_dataframe["category"],
        category_predictions,
        CATEGORY_LABELS,
        "Category Confusion Matrix",
        OUTPUTS_DIR / "category_confusion_matrix.png",
    )
    save_confusion_matrix(
        test_dataframe["urgency"],
        urgency_predictions,
        URGENCY_LABELS,
        "Urgency Confusion Matrix",
        OUTPUTS_DIR / "urgency_confusion_matrix.png",
    )
    (OUTPUTS_DIR / "classification_reports.txt").write_text(build_reports(models, test_dataframe), encoding="utf-8")
    create_error_analysis(best_model, test_dataframe, OUTPUTS_DIR / "error_analysis.csv")
    return comparison


def main() -> None:
    """Run evaluation from the command line."""

    comparison = run_evaluation()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
