# Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""Inference helpers for the Campus Triage application.

Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from campus_triage.config import CATEGORY_LABELS, ROUTING_RECOMMENDATIONS, URGENCY_LABELS
from campus_triage.features import keyword_explanation
from campus_triage.models import load_dual_classifier


EXAMPLE_MESSAGES = [
    "My FAFSA documents still say incomplete and tuition is due tomorrow. Can someone help?",
    "I cannot register for BIO 101 because there is a hold on my account.",
    "The portal keeps showing an error when I try to submit my housing form.",
    "I am worried about a student who said they might hurt themselves. Please call me ASAP.",
]


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "tfidf_logistic_regression.joblib"


def resolve_deployed_model_path(model_path: Path = MODEL_PATH) -> Path | None:
    """Return the trained model path from the repository root."""

    if model_path.exists():
        return model_path
    return None


def model_available(model_path: Path = MODEL_PATH) -> bool:
    """Return whether the deployed model artifact exists."""

    return resolve_deployed_model_path(model_path) is not None


def model_search_diagnostics(model_path: Path = MODEL_PATH) -> str:
    """Return the absolute model path checked during deployment."""

    return str(model_path)


def load_deployed_model(model_path: Path = MODEL_PATH) -> Any:
    """Load the deployed TF-IDF Logistic Regression model."""

    resolved_model_path = resolve_deployed_model_path(model_path)
    if resolved_model_path is None:
        raise FileNotFoundError(f"No deployed model artifact found at {model_path}")
    return load_dual_classifier(str(resolved_model_path))


def predict_message(message_text: str, model: Any | None = None) -> dict[str, Any]:
    """Predict triage fields and attach routing guidance."""

    deployed_model = model or load_deployed_model()
    prediction = deployed_model.predict_one(message_text)
    prediction["routing_recommendation"] = ROUTING_RECOMMENDATIONS[prediction["category"]]
    prediction["explanation"] = keyword_explanation(
        message_text,
        prediction["category"],
        prediction["category_confidence"],
    )
    prediction["category_scores"] = {label: prediction["category_scores"].get(label, 0.0) for label in CATEGORY_LABELS}
    prediction["urgency_scores"] = {label: prediction["urgency_scores"].get(label, 0.0) for label in URGENCY_LABELS}
    return prediction
