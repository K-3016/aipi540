"""Fine-tune DistilBERT for MindSignal mental-health triage classification."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from mindsignal_utils import LABEL_TO_ID, MODEL_NAME, validate_dataset_columns


DATA_PATH = Path("data/mental_health_triage_synthetic_dataset.csv")
MODEL_OUTPUT_DIR = Path("models/mindsignal-distilbert")
MAX_LENGTH = 160
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Make the demo more reproducible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inject_typo(text: str) -> str:
    """Create a tiny typo by swapping two neighboring characters in one word."""

    words = text.split()
    candidates = [idx for idx, word in enumerate(words) if len(word) > 4]
    if not candidates:
        return text

    word_idx = random.choice(candidates)
    word = words[word_idx]
    char_idx = random.randint(0, len(word) - 2)
    chars = list(word)
    chars[char_idx], chars[char_idx + 1] = chars[char_idx + 1], chars[char_idx]
    words[word_idx] = "".join(chars)
    return " ".join(words)


def augment_text(text: str) -> str:
    """Apply lightweight social-text augmentation used only for train examples."""

    augmented = text

    if random.random() < 0.35:
        augmented = augmented.lower()

    if random.random() < 0.25:
        augmented = inject_typo(augmented)

    if random.random() < 0.25:
        augmented = f"{augmented} {random.choice([':/', '<3', ':(', ':)'])}"

    if random.random() < 0.25:
        phrase = random.choice(["ngl", "idk", "rn", "honestly", "tbh"])
        augmented = f"{phrase}, {augmented}"

    return augmented


def build_augmented_training_set(train_df: pd.DataFrame, copies: int = 2) -> pd.DataFrame:
    """Add augmented copies of each training row while keeping original rows."""

    augmented_rows = []
    for _, row in train_df.iterrows():
        for copy_idx in range(copies):
            new_row = row.copy()
            new_row["id"] = f"{row['id']}_aug_{copy_idx + 1}"
            new_row["text"] = augment_text(str(row["text"]))
            new_row["notes"] = f"augmented_from={row['id']}"
            augmented_rows.append(new_row)

    if not augmented_rows:
        return train_df

    return pd.concat([train_df, pd.DataFrame(augmented_rows)], ignore_index=True)


def tokenize_dataset(dataset: Dataset, tokenizer):
    """Tokenize text and add numeric labels expected by Hugging Face Trainer."""

    def tokenize_batch(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tokenized["labels"] = [LABEL_TO_ID[label] for label in batch["label"]]
        return tokenized

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
    )


def compute_metrics(eval_prediction):
    """Return simple validation metrics during training."""

    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


def main() -> None:
    set_seed()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Add the CSV file and re-run training."
        )

    df = pd.read_csv(DATA_PATH)
    validate_dataset_columns(df.columns)
    df["id"] = df["id"].astype(str)

    train_df = df[df["split"] == "train"].copy()
    validation_df = df[df["split"].isin(["validation", "val", "dev"])].copy()

    if validation_df.empty:
        validation_df = df[df["split"] == "test"].copy()
        print("No validation split found; using test split for validation preview.")

    if train_df.empty:
        raise ValueError("No rows with split='train' were found.")

    train_df = build_augmented_training_set(train_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_TO_ID),
        id2label={idx: label for label, idx in LABEL_TO_ID.items()},
        label2id=LABEL_TO_ID,
    )

    train_dataset = tokenize_dataset(
        Dataset.from_pandas(train_df, preserve_index=False), tokenizer
    )
    validation_dataset = tokenize_dataset(
        Dataset.from_pandas(validation_df, preserve_index=False), tokenizer
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="models/training-checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"Saved fine-tuned model to {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
