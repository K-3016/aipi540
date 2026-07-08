# Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""Model definitions for baseline, classical ML, and transformer approaches.

Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from campus_triage.config import CATEGORY_LABELS, TRANSFORMER_MODEL_DIR, URGENCY_LABELS
from campus_triage.features import combine_text_fields


@dataclass
class DualClassifier:
    """Container for separate category and urgency classifiers."""

    category_model: Any
    urgency_model: Any
    model_name: str

    def predict(self, dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict category and urgency labels."""

        text_features = combine_text_fields(dataframe)
        return self.category_model.predict(text_features), self.urgency_model.predict(text_features)

    def predict_one(self, message_text: str) -> dict[str, Any]:
        """Predict one message with confidence scores when available."""

        dataframe = pd.DataFrame({"message_text": [message_text]})
        text_features = combine_text_fields(dataframe)
        category_prediction = self.category_model.predict(text_features)[0]
        urgency_prediction = self.urgency_model.predict(text_features)[0]
        category_scores = prediction_scores(self.category_model, text_features, CATEGORY_LABELS)
        urgency_scores = prediction_scores(self.urgency_model, text_features, URGENCY_LABELS)
        return {
            "category": category_prediction,
            "urgency": urgency_prediction,
            "category_scores": category_scores,
            "urgency_scores": urgency_scores,
            "category_confidence": category_scores.get(category_prediction, 0.0),
            "urgency_confidence": urgency_scores.get(urgency_prediction, 0.0),
        }


class TransformerTextClassifier:
    """Hugging Face text classifier wrapper with a scikit-learn-like API."""

    def __init__(self, model_dir: Path, batch_size: int = 16) -> None:
        """Load a fine-tuned transformer model and tokenizer."""

        if not model_dir.exists():
            raise FileNotFoundError(f"Transformer model directory not found: {model_dir}")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self.model_dir = model_dir
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        self.torch = torch
        self.classes_ = [self.model.config.id2label[index] for index in range(self.model.config.num_labels)]

    def predict_proba(self, text_features: pd.Series) -> np.ndarray:
        """Return class probabilities for input messages."""

        texts = [str(text) for text in text_features.tolist()]
        probability_batches = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                encoded = self.tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
                outputs = self.model(**encoded)
                probabilities = self.torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                probability_batches.append(probabilities)
        return np.vstack(probability_batches)

    def predict(self, text_features: pd.Series) -> np.ndarray:
        """Predict labels for input messages."""

        probabilities = self.predict_proba(text_features)
        label_indexes = probabilities.argmax(axis=1)
        return np.array([self.classes_[index] for index in label_indexes])


def prediction_scores(model: Any, text_features: pd.Series, labels: list[str]) -> dict[str, float]:
    """Return probability-like scores for a classifier."""

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_features)[0]
        classes = list(model.classes_)
        return {label: float(probabilities[classes.index(label)]) if label in classes else 0.0 for label in labels}
    decision_scores = model.decision_function(text_features)[0]
    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    probabilities = exp_scores / exp_scores.sum()
    classes = list(model.classes_)
    return {label: float(probabilities[classes.index(label)]) if label in classes else 0.0 for label in labels}


def build_baseline_model(train_dataframe: pd.DataFrame) -> DualClassifier:
    """Train majority-class baseline classifiers for category and urgency."""

    text_features = combine_text_fields(train_dataframe)
    category_model = DummyClassifier(strategy="most_frequent")
    urgency_model = DummyClassifier(strategy="most_frequent")
    category_model.fit(text_features, train_dataframe["category"])
    urgency_model.fit(text_features, train_dataframe["urgency"])
    return DualClassifier(category_model=category_model, urgency_model=urgency_model, model_name="majority_baseline")


def build_text_pipeline() -> Pipeline:
    """Create the TF-IDF plus Logistic Regression pipeline."""

    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None)),
        ]
    )


def build_classical_model(train_dataframe: pd.DataFrame) -> DualClassifier:
    """Train TF-IDF Logistic Regression classifiers for category and urgency."""

    text_features = combine_text_fields(train_dataframe)
    category_model = build_text_pipeline()
    urgency_model = build_text_pipeline()
    category_model.fit(text_features, train_dataframe["category"])
    urgency_model.fit(text_features, train_dataframe["urgency"])
    return DualClassifier(category_model=category_model, urgency_model=urgency_model, model_name="tfidf_logistic_regression")


def load_transformer_dual_classifier(model_dir: Path = TRANSFORMER_MODEL_DIR) -> DualClassifier:
    """Load fine-tuned transformer category and urgency classifiers."""

    return DualClassifier(
        category_model=TransformerTextClassifier(model_dir / "category"),
        urgency_model=TransformerTextClassifier(model_dir / "urgency"),
        model_name="distilbert_transformer",
    )


def transformer_model_available(model_dir: Path = TRANSFORMER_MODEL_DIR) -> bool:
    """Return whether fine-tuned transformer checkpoints exist."""

    category_config = model_dir / "category" / "config.json"
    urgency_config = model_dir / "urgency" / "config.json"
    return category_config.exists() and urgency_config.exists()


def save_dual_classifier(model: DualClassifier, path: str) -> None:
    """Persist a dual classifier with joblib."""

    joblib.dump(model, path)


def load_dual_classifier(path: str) -> DualClassifier:
    """Load a persisted dual classifier."""

    return joblib.load(path)


def transformer_dependencies_available() -> bool:
    """Return whether optional transformer dependencies can be imported."""

    try:
        import datasets  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True
