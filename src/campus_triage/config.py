# Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""Configuration constants for the campus triage project.

Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATA_PATH = RAW_DATA_DIR / "campus_support_messages.csv"
TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test.csv"

CLASSICAL_MODEL_PATH = MODELS_DIR / "tfidf_logistic_regression.joblib"
BASELINE_MODEL_PATH = MODELS_DIR / "majority_baseline.joblib"
TRANSFORMER_MODEL_DIR = MODELS_DIR / "transformer"

CATEGORY_LABELS = [
    "financial_aid",
    "registration",
    "housing",
    "academic_advising",
    "technical_support",
    "health_wellness",
    "general",
]

URGENCY_LABELS = ["low", "medium", "high"]

RANDOM_SEED = 42
DEFAULT_DATASET_SIZE = 1800
TEST_SIZE = 0.15
VAL_SIZE = 0.15

ROUTING_RECOMMENDATIONS = {
    "financial_aid": "Route to Financial Aid queue; attach billing or FAFSA context if available.",
    "registration": "Route to Registrar or Enrollment Services for schedule and hold review.",
    "housing": "Route to Housing and Residence Life for assignment or facilities follow-up.",
    "academic_advising": "Route to Academic Advising; include program, course, and deadline details.",
    "technical_support": "Route to IT Help Desk; request screenshots, device, and browser details.",
    "health_wellness": "Route to Student Health and Wellness; escalate immediately for safety concerns.",
    "general": "Route to general student services intake for manual triage.",
}
