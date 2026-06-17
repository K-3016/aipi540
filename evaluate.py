"""Evaluate the MindSignal classifier on test and stress-test splits."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.metrics import ConfusionMatrixDisplay

from mindsignal_utils import LABELS, load_model_and_tokenizer, predict_text, validate_dataset_columns


DATA_PATH = Path("data/mental_health_triage_synthetic_dataset.csv")
RESULTS_DIR = Path("results")
REPORT_PATH = RESULTS_DIR / "evaluation_report.txt"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"


def predict_many(texts, tokenizer, model):
    """Predict a list of messages with the rule-based safety override enabled."""

    predictions = []
    confidences = []
    override_count = 0

    for text in texts:
        result = predict_text(str(text), tokenizer, model)
        predictions.append(result.label)
        confidences.append(result.confidence)
        if result.used_safety_override:
            override_count += 1

    return predictions, confidences, override_count


def metrics_block(name: str, y_true, y_pred, override_count: int) -> str:
    """Format metrics for one split."""

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {
            "label": LABELS,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    escalation_recall = per_class.loc[
        per_class["label"] == "escalation_required", "recall"
    ].iloc[0]

    return "\n".join(
        [
            f"=== {name} split ===",
            f"rows: {len(y_true)}",
            f"accuracy: {accuracy:.4f}",
            f"macro_f1: {macro_f1:.4f}",
            f"escalation_required_recall: {escalation_recall:.4f}",
            f"safety_override_predictions: {override_count}",
            "",
            "Per-class precision/recall/F1:",
            per_class.to_string(index=False),
            "",
            "Classification report:",
            classification_report(y_true, y_pred, labels=LABELS, zero_division=0),
        ]
    )


def save_confusion_matrix(y_true, y_pred) -> None:
    """Save the confusion matrix for the standard test split."""

    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABELS)
    fig, ax = plt.subplots(figsize=(9, 7))
    display.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=30)
    ax.set_title("MindSignal Test Split Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=180)
    plt.close(fig)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Add the CSV file and re-run evaluation."
        )

    df = pd.read_csv(DATA_PATH)
    validate_dataset_columns(df.columns)

    tokenizer, model = load_model_and_tokenizer()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report_sections = [
        "MindSignal Evaluation Report",
        "Model: distilbert-base-uncased fine-tuned for 3-class triage",
        "Safety override: enabled before model prediction",
        "",
    ]

    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        raise ValueError("No rows with split='test' were found.")

    test_predictions, _, test_override_count = predict_many(test_df["text"], tokenizer, model)
    report_sections.append(
        metrics_block("test", test_df["label"], test_predictions, test_override_count)
    )
    save_confusion_matrix(test_df["label"], test_predictions)

    stress_df = df[df["split"] == "stress_test"].copy()
    if not stress_df.empty:
        stress_predictions, _, stress_override_count = predict_many(
            stress_df["text"], tokenizer, model
        )
        stress_accuracy = accuracy_score(stress_df["label"], stress_predictions)
        report_sections.extend(
            [
                "",
                metrics_block(
                    "stress_test",
                    stress_df["label"],
                    stress_predictions,
                    stress_override_count,
                ),
                f"stress_test_accuracy: {stress_accuracy:.4f}",
            ]
        )
    else:
        report_sections.append("\nNo stress_test rows were found.")

    REPORT_PATH.write_text("\n".join(report_sections), encoding="utf-8")
    print(f"Saved evaluation report to {REPORT_PATH}")
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
