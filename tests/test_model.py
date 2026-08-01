"""Tests for model-independent interfaces and mocked inference."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_features import format_inference_prompt, format_training_prompt
from scripts.model import prediction_record, validate_lora_config, validate_model_path


def test_prompt_formatting() -> None:
    prompt = format_inference_prompt("Mild cardiomegaly.")
    assert "### Medical statement" in prompt
    assert "Mild cardiomegaly." in prompt
    assert prompt.endswith("### Patient-friendly explanation\n")
    training = format_training_prompt("Mild cardiomegaly.", "The heart is slightly enlarged.")
    assert training.endswith("The heart is slightly enlarged.")


@pytest.mark.parametrize(
    ("rank", "alpha", "dropout"),
    [(0, 16, 0.05), (8, 0, 0.05), (8, 16, -0.1), (8, 16, 1.0)],
)
def test_lora_configuration_validation_rejects_invalid_values(
    rank: int, alpha: int, dropout: float
) -> None:
    with pytest.raises(ValueError):
        validate_lora_config(rank, alpha, dropout)


def test_prediction_output_schema() -> None:
    record = {
        "example_id": "example-1",
        "medical_text": "Technical text.",
        "reference_output": "Plain text.",
    }
    output = prediction_record(record, "Generated text.", "baseline_output")
    assert set(output) == {"example_id", "medical_text", "reference_output", "baseline_output"}


def test_empty_prompt_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        format_inference_prompt("   ")


def test_model_path_validation(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_model_path(tmp_path / "missing")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(ValueError, match="adapter_config.json"):
        validate_model_path(adapter)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    validate_model_path(adapter)

