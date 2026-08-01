"""Tests for evaluation metrics, heuristics, and output contracts."""

from __future__ import annotations

import math

import pytest

from scripts.model import calculate_rouge, readability_metrics, safety_heuristics


def test_readability_metric_calculation() -> None:
    metrics = readability_metrics("The heart is slightly larger than usual.")
    assert metrics["word_count"] == 7
    assert metrics["sentence_count"] == 1
    assert set(metrics) == {
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "word_count",
        "sentence_count",
    }
    assert isinstance(metrics["flesch_reading_ease"], float)
    assert math.isfinite(metrics["flesch_reading_ease"])
    assert math.isfinite(metrics["flesch_kincaid_grade"])


def test_readability_empty_input() -> None:
    metrics = readability_metrics("")
    assert metrics["word_count"] == 0
    assert math.isnan(metrics["flesch_kincaid_grade"])


def test_rouge_input_handling() -> None:
    with pytest.raises(ValueError):
        calculate_rouge([], [])
    with pytest.raises(ValueError):
        calculate_rouge(["one"], ["one", "two"])


def test_uncertainty_preservation_check() -> None:
    unsafe = safety_heuristics(
        "The finding may represent cancer.",
        "You have cancer.",
    )
    safe = safety_heuristics(
        "The finding may represent cancer.",
        "The finding may be related to cancer, but more evaluation is needed.",
    )
    assert unsafe["loss_of_uncertainty"]
    assert unsafe["unsupported_certainty"]
    assert not safe["loss_of_uncertainty"]


def test_negation_change_heuristic() -> None:
    flags = safety_heuristics(
        "No invasive cancer is identified.",
        "Invasive cancer was identified.",
    )
    assert flags["negation_change"]


def test_evaluation_flag_schema() -> None:
    flags = safety_heuristics("No lung mass is seen.", "A lung mass is seen.")
    assert set(flags) == {
        "unsupported_certainty",
        "loss_of_uncertainty",
        "new_diagnosis_language",
        "new_treatment_recommendation",
        "missing_numbers",
        "missing_anatomy",
        "negation_change",
    }
