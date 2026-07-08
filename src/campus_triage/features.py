# Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""Feature engineering helpers for support message models.

Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""

from __future__ import annotations

import re

import pandas as pd


TEXT_COLUMN = "message_text"


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving meaningful punctuation."""

    return re.sub(r"\s+", " ", str(text)).strip()


def combine_text_fields(dataframe: pd.DataFrame) -> pd.Series:
    """Return the text field used by all classifiers."""

    return dataframe[TEXT_COLUMN].fillna("").map(normalize_text)


def keyword_explanation(message_text: str, category: str, confidence: float) -> str:
    """Create a short human-readable explanation for an inference result."""

    keyword_map = {
        "financial_aid": ["aid", "fafsa", "bill", "scholarship", "payment", "loan"],
        "registration": ["register", "waitlist", "hold", "drop", "schedule", "section"],
        "housing": ["housing", "dorm", "roommate", "room", "residence", "work order"],
        "academic_advising": ["advisor", "major", "graduate", "credits", "requirement", "course"],
        "technical_support": ["login", "password", "portal", "email", "error", "canvas"],
        "health_wellness": ["health", "counseling", "anxiety", "sick", "hurt", "wellness"],
        "general": ["help", "question", "office", "information", "policy"],
    }
    lowered_text = message_text.lower()
    matches = [word for word in keyword_map.get(category, []) if word in lowered_text]
    if matches:
        return f"Matched routing cues: {', '.join(matches[:3])}. Model confidence is {confidence:.0%}."
    if confidence < 0.55:
        return "The model is uncertain; this should be reviewed by an intake specialist."
    return f"The message language is most similar to prior {category.replace('_', ' ')} examples."
