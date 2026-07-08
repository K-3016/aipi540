# Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""Synthetic data generation and train/validation/test splitting.

Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from campus_triage.config import (
    CATEGORY_LABELS,
    DEFAULT_DATASET_SIZE,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    RAW_DATA_PATH,
    TEST_PATH,
    TEST_SIZE,
    TRAIN_PATH,
    URGENCY_LABELS,
    VAL_PATH,
    VAL_SIZE,
)


@dataclass(frozen=True)
class MessageTemplate:
    """Template bank entry for a support category."""

    category: str
    phrases: tuple[str, ...]
    keywords: tuple[str, ...]


CATEGORY_TEMPLATES = [
    MessageTemplate(
        "financial_aid",
        (
            "My aid package still has not posted and tuition is due {time_phrase}.",
            "Can someone explain why my scholarship disappeared from the bill?",
            "I uploaded FAFSA documents but the portal still says incomplete.",
            "I need help setting up a payment plan before classes start.",
        ),
        ("FAFSA", "grant", "loan", "bill", "scholarship", "refund", "payment"),
    ),
    MessageTemplate(
        "registration",
        (
            "I cannot register for {course} because there is a hold on my account.",
            "The waitlist says open but the system will not let me add the class.",
            "I need to drop a course before the deadline but the button is missing.",
            "My schedule has the wrong lab section and I need it fixed.",
        ),
        ("register", "waitlist", "hold", "drop", "add", "schedule", "section"),
    ),
    MessageTemplate(
        "housing",
        (
            "My roommate assignment changed and I never received an explanation.",
            "There is no hot water in my dorm and the work order is still pending.",
            "I need to request housing accommodation documentation review.",
            "Can I move rooms because the noise is affecting my sleep?",
        ),
        ("dorm", "roommate", "meal plan", "residence", "housing", "move-in", "work order"),
    ),
    MessageTemplate(
        "academic_advising",
        (
            "I am not sure which requirement {course} satisfies for my major.",
            "Can an advisor check whether I am on track to graduate?",
            "I need approval for an overload because this is my final semester.",
            "I want to change majors and need to know the next step.",
        ),
        ("advisor", "major", "degree audit", "graduation", "credits", "requirement", "overload"),
    ),
    MessageTemplate(
        "technical_support",
        (
            "My campus login keeps failing even after I reset my password.",
            "The learning platform will not load my quiz and it is due {time_phrase}.",
            "I cannot access email from my phone after the security update.",
            "The portal shows an error code when I try to submit the form.",
        ),
        ("login", "password", "portal", "MFA", "Canvas", "email", "error"),
    ),
    MessageTemplate(
        "health_wellness",
        (
            "I need an appointment because I have been feeling overwhelmed all week.",
            "Can I talk to someone today about anxiety and missing classes?",
            "I tested positive and need to know what to do about attendance.",
            "I am worried about a student who said they might hurt themselves.",
        ),
        ("counseling", "health", "anxiety", "sick", "wellness", "urgent care", "safety"),
    ),
    MessageTemplate(
        "general",
        (
            "Hi, I am not sure who to contact about this question.",
            "Can you point me to the right office for a campus policy issue?",
            "I have a question about student services and could use guidance.",
            "Please let me know where this request should go.",
        ),
        ("question", "office", "campus", "help", "policy", "services", "information"),
    ),
]

URGENT_PHRASES = {
    "low": ("when you have time", "not urgent", "next week is fine", "just checking"),
    "medium": ("soon", "before Friday", "today if possible", "I am stuck"),
    "high": ("right now", "emergency", "deadline is tonight", "I might lose access", "please call me ASAP"),
}

COURSES = ("BIO 101", "MATH 220", "ENG 201", "CS 150", "CHEM 110", "HIST 305")
CHANNELS = ("email", "chat", "web_form")
STUDENT_TYPES = ("undergraduate", "graduate", "international", "online")
TYPO_REPLACEMENTS = {"the": "teh", "please": "plz", "because": "bc", "account": "acount", "receive": "recieve"}


def ensure_data_directories() -> None:
    """Create data directories required by the project."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def choose_urgency(category: str, random_state: random.Random) -> str:
    """Sample urgency with category-specific imbalance."""

    if category == "health_wellness":
        return random_state.choices(URGENCY_LABELS, weights=[0.25, 0.35, 0.40], k=1)[0]
    if category in {"financial_aid", "technical_support", "registration"}:
        return random_state.choices(URGENCY_LABELS, weights=[0.35, 0.45, 0.20], k=1)[0]
    return random_state.choices(URGENCY_LABELS, weights=[0.55, 0.35, 0.10], k=1)[0]


def inject_noise(text: str, random_state: random.Random) -> str:
    """Add realistic student-message noise such as typos and informal punctuation."""

    noisy_text = text
    if random_state.random() < 0.18:
        word, replacement = random_state.choice(list(TYPO_REPLACEMENTS.items()))
        noisy_text = noisy_text.replace(word, replacement)
    if random_state.random() < 0.12 and len(noisy_text) > 20:
        index = random_state.randrange(5, len(noisy_text) - 5)
        noisy_text = noisy_text[:index] + noisy_text[index + 1 :]
    if random_state.random() < 0.18:
        noisy_text = noisy_text.lower()
    if random_state.random() < 0.16:
        noisy_text += random_state.choice(("!!", "???", " pls", " thx", " :/"))
    if random_state.random() < 0.10:
        noisy_text = noisy_text.replace(".", "")
    return noisy_text


def build_message(template: MessageTemplate, urgency: str, random_state: random.Random) -> str:
    """Create one synthetic support message from templates and slots."""

    phrase = random_state.choice(template.phrases)
    message = phrase.format(
        course=random_state.choice(COURSES),
        time_phrase=random_state.choice(("tomorrow", "tonight", "this week", "in two days")),
    )
    urgency_phrase = random_state.choice(URGENT_PHRASES[urgency])
    keyword = random_state.choice(template.keywords)
    if random_state.random() < 0.35:
        message = f"{message} Also, the {keyword} page is confusing."
    if random_state.random() < 0.28:
        message = f"{message} {urgency_phrase}."
    if random_state.random() < 0.12:
        message = random_state.choice(("help", "need help asap", "confused about this", message))
    if random_state.random() < 0.20:
        message = f"Hi team, {message} I already checked the student portal and could not find a clear answer."
    return inject_noise(message, random_state)


def generate_synthetic_dataset(row_count: int = DEFAULT_DATASET_SIZE, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a synthetic campus support triage dataset."""

    random_state = random.Random(seed)
    rows = []
    category_weights = [0.17, 0.16, 0.13, 0.16, 0.15, 0.10, 0.13]
    for index in range(row_count):
        template = random_state.choices(CATEGORY_TEMPLATES, weights=category_weights, k=1)[0]
        urgency = choose_urgency(template.category, random_state)
        rows.append(
            {
                "message_id": f"MSG-{index + 1:05d}",
                "message_text": build_message(template, urgency, random_state),
                "category": template.category,
                "urgency": urgency,
                "channel": random_state.choice(CHANNELS),
                "student_type": random_state.choice(STUDENT_TYPES),
                "created_hour": random_state.randrange(0, 24),
            }
        )
    return pd.DataFrame(rows)


def stratification_key(dataframe: pd.DataFrame) -> pd.Series:
    """Build a combined category/urgency key for stratified splitting."""

    return dataframe["category"].astype(str) + "__" + dataframe["urgency"].astype(str)


def safe_train_test_split(dataframe: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split with stratification when class counts permit it."""

    stratify = stratification_key(dataframe)
    if stratify.value_counts().min() < 2:
        stratify = None
    return train_test_split(dataframe, test_size=test_size, random_state=seed, stratify=stratify)


def save_dataset_splits(dataframe: pd.DataFrame, seed: int = RANDOM_SEED) -> None:
    """Save raw, train, validation, and test CSV files."""

    ensure_data_directories()
    dataframe.to_csv(RAW_DATA_PATH, index=False)
    train_val, test = safe_train_test_split(dataframe, TEST_SIZE, seed)
    relative_val_size = VAL_SIZE / (1.0 - TEST_SIZE)
    train, validation = safe_train_test_split(train_val, relative_val_size, seed)
    train.to_csv(TRAIN_PATH, index=False)
    validation.to_csv(VAL_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)


def load_processed_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test dataframes."""

    return pd.read_csv(TRAIN_PATH), pd.read_csv(VAL_PATH), pd.read_csv(TEST_PATH)


def create_and_save_dataset(row_count: int = DEFAULT_DATASET_SIZE, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate and persist all dataset files."""

    dataframe = generate_synthetic_dataset(row_count=row_count, seed=seed)
    save_dataset_splits(dataframe, seed=seed)
    return dataframe


def validate_dataset_columns(dataframe: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Raise a clear error if required dataset columns are missing."""

    missing_columns = set(required_columns) - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
