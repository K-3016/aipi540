# Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""Robustness experiment with noisy text transformations.

Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from campus_triage.config import CLASSICAL_MODEL_PATH, OUTPUTS_DIR, RANDOM_SEED, TEST_PATH
from campus_triage.evaluate import evaluate_dual_classifier, ensure_models_and_data
from campus_triage.models import (
    DualClassifier,
    load_dual_classifier,
    load_transformer_dual_classifier,
    transformer_model_available,
)


def delete_random_character(text: str, random_state: random.Random) -> str:
    """Delete one random non-space character from text."""

    candidates = [index for index, char in enumerate(text) if not char.isspace()]
    if not candidates:
        return text
    index = random_state.choice(candidates)
    return text[:index] + text[index + 1 :]


def add_random_typo(text: str, random_state: random.Random) -> str:
    """Swap two adjacent characters in a random word."""

    words = text.split()
    eligible = [index for index, word in enumerate(words) if len(word) > 4]
    if not eligible:
        return text
    word_index = random_state.choice(eligible)
    word = words[word_index]
    char_index = random_state.randrange(0, len(word) - 1)
    words[word_index] = word[:char_index] + word[char_index + 1] + word[char_index] + word[char_index + 2 :]
    return " ".join(words)


def make_text_noisy(text: str, random_state: random.Random) -> str:
    """Apply deletion, typo, lowercasing, extra punctuation, and missing punctuation."""

    noisy_text = delete_random_character(str(text), random_state)
    noisy_text = add_random_typo(noisy_text, random_state)
    noisy_text = noisy_text.lower()
    noisy_text = noisy_text.replace(".", "")
    noisy_text = noisy_text + random_state.choice(["!!!", "???", " ...", ""])
    return noisy_text


def make_noisy_dataframe(dataframe: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create a noisy copy of a dataset."""

    random_state = random.Random(seed)
    noisy_dataframe = dataframe.copy()
    noisy_dataframe["message_text"] = noisy_dataframe["message_text"].map(lambda text: make_text_noisy(text, random_state))
    return noisy_dataframe


def evaluate_clean_and_noisy(model: DualClassifier, clean_dataframe: pd.DataFrame, noisy_dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Evaluate one model on clean and noisy test sets."""

    clean_results = evaluate_dual_classifier(model, clean_dataframe)
    clean_results["condition"] = "clean"
    noisy_results = evaluate_dual_classifier(model, noisy_dataframe)
    noisy_results["condition"] = "noisy"
    return [clean_results, noisy_results]


def save_robustness_plot(results: pd.DataFrame, output_path: Path) -> None:
    """Save a bar plot comparing clean and noisy macro F1."""

    plot_dataframe = results.melt(
        id_vars=["model", "condition"],
        value_vars=["category_macro_f1", "urgency_macro_f1"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 5))
    labels = [f"{row.model}\n{row.condition}\n{row.metric.replace('_macro_f1', '')}" for row in plot_dataframe.itertuples()]
    plt.bar(range(len(plot_dataframe)), plot_dataframe["score"], color="#0f766e")
    plt.xticks(range(len(plot_dataframe)), labels, rotation=30, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Macro F1")
    plt.title("Robustness to Noisy Student Messages")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_robustness_experiment() -> pd.DataFrame:
    """Compare classical and transformer models on clean/noisy data when available."""

    ensure_models_and_data()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    test_dataframe = pd.read_csv(TEST_PATH)
    noisy_dataframe = make_noisy_dataframe(test_dataframe)
    models = [load_dual_classifier(str(CLASSICAL_MODEL_PATH))]
    if transformer_model_available():
        models.append(load_transformer_dual_classifier())

    rows = []
    for model in models:
        rows.extend(evaluate_clean_and_noisy(model, test_dataframe, noisy_dataframe))

    if not transformer_model_available():
        rows.append(
            {
                "model": "distilbert_transformer",
                "condition": "not_trained",
                "category_accuracy": None,
                "category_macro_f1": None,
                "category_weighted_f1": None,
                "urgency_accuracy": None,
                "urgency_macro_f1": None,
                "urgency_weighted_f1": None,
                "interpretation": "Deep learning implementation is in src/campus_triage/transformer_training.py. Run `make train-transformer` to add transformer clean/noisy scores.",
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUTS_DIR / "robustness_experiment.csv", index=False)
    plottable_results = results.dropna(subset=["category_macro_f1", "urgency_macro_f1"])
    save_robustness_plot(plottable_results, OUTPUTS_DIR / "robustness_plot.png")
    return results


def main() -> None:
    """Run robustness experiment from the command line."""

    results = run_robustness_experiment()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
