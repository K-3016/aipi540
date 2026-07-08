# Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""Optional lightweight DistilBERT fine-tuning utilities.

Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""

from __future__ import annotations

from pathlib import Path
import inspect

import pandas as pd

from campus_triage.config import CATEGORY_LABELS, URGENCY_LABELS


def train_single_transformer(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    label_column: str,
    labels: list[str],
    output_dir: Path,
    max_rows: int,
) -> None:
    """Fine-tune one DistilBERT sequence classifier."""

    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    label_to_id = {label: index for index, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_subset = train_dataframe.sample(min(max_rows, len(train_dataframe)), random_state=42).copy()
    validation_subset = validation_dataframe.copy()
    train_subset["label"] = train_subset[label_column].map(label_to_id)
    validation_subset["label"] = validation_subset[label_column].map(label_to_id)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[int]]:
        return tokenizer(batch["message_text"], truncation=True, padding="max_length", max_length=128)

    train_dataset = Dataset.from_pandas(train_subset[["message_text", "label"]]).map(tokenize, batched=True)
    validation_dataset = Dataset.from_pandas(validation_subset[["message_text", "label"]]).map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(labels),
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
    )
    training_args_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": 1,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "save_strategy": "epoch",
        "logging_steps": 20,
        "report_to": [],
    }
    training_args_signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in training_args_signature.parameters:
        training_args_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in training_args_signature.parameters:
        training_args_kwargs["eval_strategy"] = "epoch"
    training_args = TrainingArguments(**training_args_kwargs)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def train_transformer_models(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    output_dir: Path,
    max_rows: int = 600,
) -> None:
    """Train category and urgency transformer classifiers."""

    train_single_transformer(
        train_dataframe,
        validation_dataframe,
        "category",
        CATEGORY_LABELS,
        output_dir / "category",
        max_rows,
    )
    train_single_transformer(
        train_dataframe,
        validation_dataframe,
        "urgency",
        URGENCY_LABELS,
        output_dir / "urgency",
        max_rows,
    )
