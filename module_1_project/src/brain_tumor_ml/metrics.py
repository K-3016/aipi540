from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .constants import CLASS_NAMES


def classification_metrics(y_true, probabilities) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"Expected probabilities with shape (n, {len(CLASS_NAMES)}).")
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    labels = np.arange(len(CLASS_NAMES))
    matrix = confusion_matrix(y_true, predictions, labels=labels)
    one_hot = np.eye(len(CLASS_NAMES))[y_true]
    per_class = {}
    for index, class_name in enumerate(CLASS_NAMES):
        tp = matrix[index, index]
        fn = matrix[index].sum() - tp
        fp = matrix[:, index].sum() - tp
        tn = matrix.sum() - tp - fn - fp
        per_class[class_name] = {
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "sensitivity_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "f1": float(
                f1_score(y_true == index, predictions == index, zero_division=0)
            ),
            "support": int((y_true == index).sum()),
        }
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_precision": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, predictions, average="weighted", zero_division=0)
        ),
        "roc_auc_ovr_macro": _safe_score(
            roc_auc_score, y_true, probabilities, multi_class="ovr", average="macro"
        ),
        "log_loss": _safe_score(log_loss, y_true, probabilities, labels=labels),
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class": per_class,
        "confidence_triage": confidence_triage_metrics(y_true, probabilities),
        "n_samples": int(len(y_true)),
    }


def confidence_triage_metrics(
    y_true,
    probabilities,
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> list[dict]:
    """Measure accuracy and coverage when low-confidence cases are sent for review."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    rows = []
    for threshold in thresholds:
        retained = confidence >= threshold
        count = int(retained.sum())
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": float(retained.mean()),
                "retained_cases": count,
                "review_cases": int(len(y_true) - count),
                "selective_accuracy": (
                    float(accuracy_score(y_true[retained], predictions[retained]))
                    if count
                    else None
                ),
            }
        )
    return rows


def _safe_score(function, y_true, probabilities, **kwargs) -> float | None:
    try:
        return float(function(y_true, probabilities, **kwargs))
    except ValueError:
        return None
