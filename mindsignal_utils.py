"""Shared helpers for the MindSignal prototype."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABELS = [
    "informational",
    "emotional_support_needed",
    "escalation_required",
]

LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

MODEL_NAME = "distilbert-base-uncased"
DEFAULT_MODEL_DIR = Path("models/mindsignal-distilbert")

HIGH_RISK_PHRASES = [
    "kill myself",
    "end my life",
    "not worth living",
    "hurt myself",
    "suicide",
    "not safe",
    "better off without me",
    "can't keep myself safe",
    "cant keep myself safe",
]


@dataclass
class Prediction:
    """A simple prediction result used by both evaluation and Streamlit."""

    label: str
    confidence: float
    used_safety_override: bool


def validate_dataset_columns(columns: Iterable[str]) -> None:
    """Raise a helpful error if the CSV does not have the expected columns."""

    required = {"id", "text", "label", "test_type", "split", "notes"}
    missing = required.difference(columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_list}")


def safety_override_label(text: str) -> str | None:
    """Return escalation_required when high-risk wording is detected."""

    normalized_text = text.lower()
    for phrase in HIGH_RISK_PHRASES:
        if phrase in normalized_text:
            return "escalation_required"
    return None


def load_model_and_tokenizer(model_dir: Path = DEFAULT_MODEL_DIR):
    """Load the fine-tuned model if present, otherwise explain what to run."""

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. Run `python train.py` first."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model


def predict_text(
    text: str,
    tokenizer,
    model,
    max_length: int = 160,
    device: str | torch.device | None = None,
) -> Prediction:
    """Classify one text message with a safety override before model inference."""

    override = safety_override_label(text)
    if override is not None:
        return Prediction(
            label=override,
            confidence=1.0,
            used_safety_override=True,
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        predicted_id = int(torch.argmax(probabilities).item())

    return Prediction(
        label=ID_TO_LABEL[predicted_id],
        confidence=float(probabilities[predicted_id].item()),
        used_safety_override=False,
    )
