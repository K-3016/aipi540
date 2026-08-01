"""Model training, inference, evaluation, and safety utilities for MedExplain."""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from scripts.build_features import format_inference_prompt
    from scripts.make_dataset import load_jsonl
except ModuleNotFoundError:  # Supports `python scripts/model.py`.
    from build_features import format_inference_prompt
    from make_dataset import load_jsonl

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER = Path("models/medexplain_lora_adapter")
DISCLAIMER = (
    "This prototype explains medical terminology for educational purposes. It does not "
    "provide medical advice, diagnosis, or treatment recommendations. AI-generated "
    "explanations may contain errors and should be reviewed by a qualified healthcare professional."
)
UNCERTAINTY_TERMS = (
    "possible",
    "possibly",
    "may",
    "might",
    "suggests",
    "concerning for",
    "cannot exclude",
    "likely",
    "unlikely",
    "suspicious for",
)
NEGATION_TERMS = ("no", "not", "without", "negative", "absent", "neither", "never")
TREATMENT_TERMS = (
    "take ",
    "start ",
    "stop ",
    "prescribe",
    "surgery",
    "chemotherapy",
    "radiation therapy",
    "dose",
)
DIAGNOSIS_ASSERTIONS = ("you have ", "this is cancer", "diagnosed with", "definitely ")
ANATOMY_TERMS = (
    "lung", "heart", "liver", "kidney", "brain", "breast", "bone", "ankle",
    "wrist", "knee", "uterus", "thyroid", "blood", "left", "right",
)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch when installed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        LOGGER.warning("Torch is not installed; only non-model utilities are available.")


def detect_device() -> str:
    """Return the best available Transformers device label."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def load_tokenizer(model_name: str = DEFAULT_MODEL) -> Any:
    """Load and normalize a Hugging Face tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(
    model_name: str = DEFAULT_MODEL,
    use_4bit: bool = False,
    for_training: bool = False,
) -> Any:
    """Load the base causal LM with optional CUDA-only 4-bit quantization."""
    import torch
    from transformers import AutoModelForCausalLM

    device = detect_device()
    kwargs: dict[str, Any] = {"trust_remote_code": False}
    if use_4bit:
        cpu_backend = device == "cpu" and platform.system() == "Linux"
        if device != "cuda" and not cpu_backend:
            raise ValueError(
                "4-bit loading requires CUDA or the bitsandbytes Linux CPU backend."
            )
        try:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(
                    torch.bfloat16
                    if device == "cuda" and torch.cuda.is_bf16_supported()
                    else torch.float16
                    if device == "cuda"
                    else torch.float32
                ),
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"
        except ImportError as exc:
            raise RuntimeError("Install bitsandbytes for --use-4bit training.") from exc
    elif device == "cuda":
        kwargs.update(
            device_map="auto",
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    elif device == "mps":
        kwargs["dtype"] = torch.float16
    else:
        kwargs["dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device == "mps":
        model = model.to("mps")
    LOGGER.info("Loaded %s on %s", model_name, device.upper())
    if device == "cpu" and not use_4bit:
        LOGGER.warning(
            "No CUDA or MPS accelerator was detected. Baseline generation can take "
            "many minutes and LoRA training can take hours on CPU."
        )
    elif device == "cpu" and use_4bit:
        LOGGER.info("Using 4-bit bitsandbytes CPU loading for constrained deployment.")
    if for_training:
        model.config.use_cache = False
    return model


def identify_lora_target_modules(model: Any) -> list[str]:
    """Identify supported projection layer names from the loaded architecture."""
    common = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "query_key_value", "dense_h_to_4h", "dense_4h_to_h",
        "c_attn", "c_proj",
    }
    found = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if module.__class__.__name__ in {"Linear", "Linear4bit"} and name.rsplit(".", 1)[-1] in common
    }
    preferred = [name for name in ("q_proj", "k_proj", "v_proj", "o_proj") if name in found]
    targets = preferred or sorted(found)
    if not targets:
        raise ValueError(
            "Could not identify LoRA target modules. Inspect model.named_modules() "
            "and configure architecture-specific linear projections."
        )
    return targets


def validate_lora_config(r: int, alpha: int, dropout: float) -> None:
    """Validate LoRA hyperparameters without loading a model."""
    if r <= 0:
        raise ValueError("LoRA rank r must be positive.")
    if alpha <= 0:
        raise ValueError("lora_alpha must be positive.")
    if not 0 <= dropout < 1:
        raise ValueError("lora_dropout must be in [0, 1).")


def configure_lora(
    model: Any,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    use_4bit: bool = False,
) -> Any:
    """Attach a PEFT LoRA adapter and report trainable parameters."""
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    validate_lora_config(r, lora_alpha, lora_dropout)
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=identify_lora_target_modules(model),
    )
    model = get_peft_model(model, config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info("Trainable parameters: %s / %s (%.4f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)
    return model


def _tokenize_dataset(records: list[dict[str, Any]], tokenizer: Any, max_length: int) -> Any:
    """Convert JSON records into a Hugging Face Dataset with causal-LM labels."""
    from datasets import Dataset

    dataset = Dataset.from_list([{"text": row["text"]} for row in records])

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        encoded = tokenizer(batch["text"], truncation=True, max_length=max_length)
        encoded["labels"] = [ids.copy() for ids in encoded["input_ids"]]
        return encoded

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def train_model(
    train_path: Path = Path("data/processed/train.jsonl"),
    validation_path: Path = Path("data/processed/validation.jsonl"),
    model_name: str = DEFAULT_MODEL,
    adapter_dir: Path = DEFAULT_ADAPTER,
    epochs: float = 2.0,
    learning_rate: float = 2e-4,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_length: int = 512,
    use_4bit: bool = False,
    seed: int = 42,
) -> Path:
    """Fine-tune LoRA parameters and save only the adapter and tokenizer."""
    import torch
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    set_seed(seed)
    tokenizer = load_tokenizer(model_name)
    model = configure_lora(load_base_model(model_name, use_4bit, for_training=True), use_4bit=use_4bit)
    train_dataset = _tokenize_dataset(load_jsonl(train_path), tokenizer, max_length)
    validation_dataset = _tokenize_dataset(load_jsonl(validation_path), tokenizer, max_length)
    device = detect_device()
    args = TrainingArguments(
        output_dir=str(adapter_dir.parent / "checkpoints"),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        fp16=device == "cuda" and not torch.cuda.is_bf16_supported(),
        bf16=device == "cuda" and torch.cuda.is_bf16_supported(),
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt"),
    )
    trainer.train()
    save_adapter(model, tokenizer, adapter_dir)
    return adapter_dir


def save_adapter(model: Any, tokenizer: Any, adapter_dir: Path = DEFAULT_ADAPTER) -> None:
    """Save PEFT adapter weights and tokenizer configuration."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    LOGGER.info("Saved LoRA adapter to %s", adapter_dir)


def load_adapted_model(
    model_name: str = DEFAULT_MODEL,
    adapter_dir: Path = DEFAULT_ADAPTER,
    use_4bit: bool = False,
) -> tuple[Any, Any]:
    """Load the base model and attach an existing PEFT adapter."""
    from peft import PeftModel

    validate_model_path(adapter_dir)
    tokenizer = load_tokenizer(model_name)
    model = PeftModel.from_pretrained(load_base_model(model_name, use_4bit), adapter_dir)
    return model, tokenizer


def validate_model_path(adapter_dir: Path) -> None:
    """Require a valid adapter configuration before adapted inference."""
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found at {adapter_dir}. Run `python main.py train` first."
        )
    if not (adapter_dir / "adapter_config.json").exists():
        raise ValueError(f"{adapter_dir} is missing adapter_config.json.")


def generate_text(
    model: Any,
    tokenizer: Any,
    medical_text: str,
    max_new_tokens: int = 150,
    repetition_penalty: float = 1.1,
) -> str:
    """Generate one deterministic patient-friendly rewrite."""
    import torch

    prompt = format_inference_prompt(medical_text)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    # Some instruction models ship sampling defaults in generation_config.
    # Sampling-only values are invalid with deterministic greedy decoding.
    generation_config = model.generation_config
    generation_config.do_sample = False
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            generation_config=generation_config,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def prediction_record(record: dict[str, Any], output: str, output_column: str) -> dict[str, str]:
    """Build the stable prediction output schema."""
    return {
        "example_id": str(record["example_id"]),
        "medical_text": str(record["medical_text"]),
        "reference_output": str(record["reference_output"]),
        output_column: output,
    }


def run_inference(
    records: Sequence[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    output_path: Path,
    output_column: str,
) -> pd.DataFrame:
    """Generate predictions with progress logging and resumable CSV checkpoints."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    completed_ids: set[str] = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        expected = ["example_id", "medical_text", "reference_output", output_column]
        if set(expected).issubset(existing.columns):
            rows = existing[expected].to_dict("records")
            completed_ids = {str(value) for value in existing["example_id"]}
            LOGGER.info("Resuming %s with %d completed examples", output_column, len(completed_ids))
    total = len(records)
    LOGGER.info("Starting %s generation for %d examples", output_column, total)
    for index, record in enumerate(records, start=1):
        if str(record["example_id"]) in completed_ids:
            continue
        LOGGER.info("Generating %s example %d/%d", output_column, index, total)
        output = generate_text(model, tokenizer, str(record["medical_text"]))
        rows.append(prediction_record(record, output, output_column))
        pd.DataFrame(rows).to_csv(output_path, index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return frame


def run_baseline_inference(
    test_path: Path = Path("data/processed/test.jsonl"),
    output_path: Path = Path("data/outputs/baseline_predictions.csv"),
    model_name: str = DEFAULT_MODEL,
) -> pd.DataFrame:
    """Run base-model inference on the held-out test set."""
    tokenizer = load_tokenizer(model_name)
    model = load_base_model(model_name)
    return run_inference(load_jsonl(test_path), model, tokenizer, output_path, "baseline_output")


def run_adapted_inference(
    test_path: Path = Path("data/processed/test.jsonl"),
    output_path: Path = Path("data/outputs/finetuned_predictions.csv"),
    model_name: str = DEFAULT_MODEL,
    adapter_dir: Path = DEFAULT_ADAPTER,
) -> pd.DataFrame:
    """Run adapter-attached inference on the same held-out test set."""
    model, tokenizer = load_adapted_model(model_name, adapter_dir)
    return run_inference(load_jsonl(test_path), model, tokenizer, output_path, "finetuned_output")


def _simple_sentence_count(text: str) -> int:
    return max(1, len(re.findall(r"[.!?]+(?:\s|$)", text.strip()))) if text.strip() else 0


def _estimate_syllables(word: str) -> int:
    """Estimate English syllables without requiring an external pronunciation corpus."""
    clean = re.sub(r"[^a-z]", "", word.lower())
    if not clean:
        return 0
    groups = len(re.findall(r"[aeiouy]+", clean))
    if clean.endswith("e") and not clean.endswith(("le", "ye")) and groups > 1:
        groups -= 1
    return max(1, groups)


def _fallback_flesch_metrics(words: Sequence[str], sentence_count: int) -> tuple[float, float]:
    """Calculate approximate Flesch metrics using an offline syllable heuristic."""
    word_count = len(words)
    syllable_count = sum(_estimate_syllables(word) for word in words)
    words_per_sentence = word_count / max(1, sentence_count)
    syllables_per_word = syllable_count / max(1, word_count)
    reading_ease = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    grade_level = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    return reading_ease, grade_level


def readability_metrics(text: str) -> dict[str, float]:
    """Calculate offline readability and length metrics without corpus downloads."""
    clean = str(text or "").strip()
    words = re.findall(r"\b[\w'-]+\b", clean)
    if not words:
        return {
            "flesch_reading_ease": math.nan,
            "flesch_kincaid_grade": math.nan,
            "word_count": 0,
            "sentence_count": 0,
        }
    sentence_count = _simple_sentence_count(clean)
    # Always use the deterministic approximation. Recent textstat releases may
    # attempt to fetch NLTK's optional cmudict corpus, which is inappropriate for
    # an offline demo and can crash after an otherwise successful inference.
    ease, grade = _fallback_flesch_metrics(words, sentence_count)
    return {
        "flesch_reading_ease": ease,
        "flesch_kincaid_grade": grade,
        "word_count": len(words),
        "sentence_count": sentence_count,
    }


def _present_terms(text: str, terms: Iterable[str]) -> set[str]:
    lowered = text.lower()
    return {term for term in terms if re.search(rf"\b{re.escape(term)}\b", lowered)}


def safety_heuristics(source: str, output: str) -> dict[str, bool]:
    """Run transparent, conservative medical-safety rules (not clinical validation)."""
    source_lower, output_lower = source.lower(), output.lower()
    number_pattern = r"(?<![\w-])\d+(?:\.\d+)?(?![\w-])"
    source_numbers = set(re.findall(number_pattern, source))
    output_numbers = set(re.findall(number_pattern, output))
    source_anatomy = _present_terms(source, ANATOMY_TERMS)
    output_anatomy = _present_terms(output, ANATOMY_TERMS)
    source_uncertainty = _present_terms(source, UNCERTAINTY_TERMS)
    output_uncertainty = _present_terms(output, UNCERTAINTY_TERMS)
    source_negation = _present_terms(source, NEGATION_TERMS)
    output_negation = _present_terms(output, NEGATION_TERMS)
    return {
        "unsupported_certainty": bool(source_uncertainty and not output_uncertainty),
        "loss_of_uncertainty": bool(source_uncertainty and not output_uncertainty),
        "new_diagnosis_language": any(term in output_lower and term not in source_lower for term in DIAGNOSIS_ASSERTIONS),
        "new_treatment_recommendation": any(term in output_lower and term not in source_lower for term in TREATMENT_TERMS),
        "missing_numbers": not source_numbers.issubset(output_numbers),
        "missing_anatomy": not source_anatomy.issubset(output_anatomy),
        "negation_change": bool(source_negation != output_negation),
    }


def calculate_rouge(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """Calculate aggregate ROUGE-1, ROUGE-2, and ROUGE-L F1."""
    if len(predictions) != len(references) or not predictions:
        raise ValueError("ROUGE requires equally sized, non-empty prediction and reference lists.")
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = [scorer.score(reference, prediction) for prediction, reference in zip(predictions, references)]
        return {
            metric: float(np.mean([score[metric].fmeasure for score in scores]))
            for metric in ("rouge1", "rouge2", "rougeL")
        }
    except ImportError:
        LOGGER.warning(
            "rouge-score is unavailable; using the built-in token-overlap ROUGE "
            "fallback without stemming."
        )

    def f1(overlap: int, predicted_count: int, reference_count: int) -> float:
        if overlap == 0 or predicted_count == 0 or reference_count == 0:
            return 0.0
        precision = overlap / predicted_count
        recall = overlap / reference_count
        return 2 * precision * recall / (precision + recall)

    def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
        return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))

    def lcs_length(left: list[str], right: list[str]) -> int:
        previous = [0] * (len(right) + 1)
        for left_token in left:
            current = [0]
            for index, right_token in enumerate(right, start=1):
                current.append(
                    previous[index - 1] + 1
                    if left_token == right_token
                    else max(previous[index], current[-1])
                )
            previous = current
        return previous[-1]

    aggregates = {"rouge1": [], "rouge2": [], "rougeL": []}
    for prediction, reference in zip(predictions, references):
        predicted_tokens = re.findall(r"\b\w+\b", prediction.lower())
        reference_tokens = re.findall(r"\b\w+\b", reference.lower())
        for n, metric in ((1, "rouge1"), (2, "rouge2")):
            predicted_ngrams = ngrams(predicted_tokens, n)
            reference_ngrams = ngrams(reference_tokens, n)
            overlap = sum((predicted_ngrams & reference_ngrams).values())
            aggregates[metric].append(f1(overlap, sum(predicted_ngrams.values()), sum(reference_ngrams.values())))
        lcs = lcs_length(predicted_tokens, reference_tokens)
        aggregates["rougeL"].append(f1(lcs, len(predicted_tokens), len(reference_tokens)))
    return {metric: float(np.mean(values)) for metric, values in aggregates.items()}


def _calculate_bertscore(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    from bert_score import score

    precision, recall, f1 = score(list(predictions), list(references), lang="en", verbose=False)
    return {
        "bertscore_precision": float(precision.mean()),
        "bertscore_recall": float(recall.mean()),
        "bertscore_f1": float(f1.mean()),
    }


def classify_failures(source: str, reference: str, output: str) -> str:
    """Map heuristic flags to explicit error-analysis categories."""
    flags = safety_heuristics(source, output)
    categories: list[str] = []
    if flags["new_diagnosis_language"] or flags["new_treatment_recommendation"]:
        categories.append("hallucinated information")
    if flags["loss_of_uncertainty"]:
        categories.append("loss of diagnostic uncertainty")
    if flags["negation_change"]:
        categories.append("negation reversal")
    if flags["missing_numbers"]:
        categories.append("number or measurement alteration")
    if flags["missing_anatomy"]:
        categories.append("omission of clinically important information")
    output_words = len(output.split())
    reference_words = max(1, len(reference.split()))
    if output_words < reference_words * 0.45:
        categories.append("oversimplification")
    if not categories:
        categories.append("manual review required")
    return "; ".join(categories)


def _prefix_metrics(frame: pd.DataFrame, text_column: str, prefix: str) -> pd.DataFrame:
    metrics = frame[text_column].fillna("").map(readability_metrics).apply(pd.Series).add_prefix(f"{prefix}_")
    result = pd.concat([frame, metrics], axis=1)
    input_words = result["original_word_count"].replace(0, np.nan)
    result[f"{prefix}_output_to_input_length_ratio"] = result[f"{prefix}_word_count"] / input_words
    return result


def evaluate_outputs(
    baseline_path: Path = Path("data/outputs/baseline_predictions.csv"),
    finetuned_path: Path = Path("data/outputs/finetuned_predictions.csv"),
    output_dir: Path = Path("data/outputs"),
    skip_bertscore: bool = False,
) -> dict[str, Any]:
    """Compare both models on identical IDs and save all evaluation artifacts."""
    baseline = pd.read_csv(baseline_path)
    finetuned = pd.read_csv(finetuned_path)
    required_baseline = {"example_id", "medical_text", "reference_output", "baseline_output"}
    required_finetuned = {"example_id", "medical_text", "reference_output", "finetuned_output"}
    if not required_baseline.issubset(baseline) or not required_finetuned.issubset(finetuned):
        raise ValueError("Prediction CSV files do not have the required schema.")
    comparison = baseline.merge(
        finetuned[["example_id", "finetuned_output"]],
        on="example_id",
        how="inner",
        validate="one_to_one",
    )
    if len(comparison) != len(baseline) or set(comparison["example_id"]) != set(finetuned["example_id"]):
        raise ValueError("Baseline and fine-tuned predictions must cover exactly the same test IDs.")
    comparison = _prefix_metrics(comparison, "medical_text", "original")
    comparison = _prefix_metrics(comparison, "baseline_output", "baseline")
    comparison = _prefix_metrics(comparison, "finetuned_output", "finetuned")
    for model_column, prefix in (("baseline_output", "baseline"), ("finetuned_output", "finetuned")):
        flag_rows = [
            safety_heuristics(source, output)
            for source, output in zip(comparison["medical_text"], comparison[model_column].fillna(""))
        ]
        flags = pd.DataFrame(flag_rows).add_prefix(f"{prefix}_")
        comparison = pd.concat([comparison, flags], axis=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "comparison_results.csv", index=False)
    references = comparison["reference_output"].fillna("").tolist()
    summary: dict[str, Any] = {
        "test_examples": len(comparison),
        "notes": {
            "rouge": "Lexical overlap can penalize valid paraphrases.",
            "readability": "Lower grade level is not evidence of medical accuracy.",
            "safety": "Safety checks are transparent heuristics, not clinical validation.",
        },
        "models": {},
    }
    for column, name in (("baseline_output", "baseline"), ("finetuned_output", "finetuned")):
        predictions = comparison[column].fillna("").tolist()
        model_summary: dict[str, Any] = {
            "rouge": calculate_rouge(predictions, references),
            "mean_readability": {
                metric: float(comparison[f"{name}_{metric}"].mean())
                for metric in ("flesch_reading_ease", "flesch_kincaid_grade", "word_count", "sentence_count", "output_to_input_length_ratio")
            },
            "safety_flag_counts": {
                flag: int(comparison[f"{name}_{flag}"].sum())
                for flag in (
                    "unsupported_certainty", "loss_of_uncertainty", "new_diagnosis_language",
                    "new_treatment_recommendation", "missing_numbers", "missing_anatomy", "negation_change",
                )
            },
        }
        if skip_bertscore:
            model_summary["bertscore"] = {"status": "skipped by user"}
        else:
            model_summary["bertscore"] = _calculate_bertscore(predictions, references)
        summary["models"][name] = model_summary
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_error_analysis(comparison, output_dir)
    write_human_evaluation_template(comparison, output_dir / "human_evaluation_template.csv")
    return summary


def write_error_analysis(comparison: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Create row-level heuristic triage plus a concise markdown guide."""
    rows = []
    for row in comparison.itertuples(index=False):
        rows.append(
            {
                "example_id": row.example_id,
                "medical_text": row.medical_text,
                "reference_output": row.reference_output,
                "baseline_output": row.baseline_output,
                "finetuned_output": row.finetuned_output,
                "baseline_failure_type": classify_failures(row.medical_text, row.reference_output, str(row.baseline_output)),
                "finetuned_failure_type": classify_failures(row.medical_text, row.reference_output, str(row.finetuned_output)),
                "notes": "",
            }
        )
    errors = pd.DataFrame(rows)
    errors.to_csv(output_dir / "error_analysis.csv", index=False)
    categories = (
        "Hallucinated information", "Loss of diagnostic uncertainty", "Oversimplification",
        "Incorrect medical interpretation", "Omission of clinically important information",
        "Negation reversal", "Number or measurement alteration",
    )
    counts = Counter(
        category.strip()
        for column in ("baseline_failure_type", "finetuned_failure_type")
        for value in errors[column]
        for category in str(value).split(";")
    )
    lines = [
        "# Error Analysis",
        "",
        "This file combines automatic heuristic triage with categories that require clinician review.",
        "Absence of a flag does not establish safety or correctness.",
        "",
        "## Review categories",
        "",
    ]
    lines.extend(f"- {category}: heuristic mentions={counts.get(category.lower(), 0)}" for category in categories)
    lines.extend(
        [
            "",
            "## Review protocol",
            "",
            "A qualified reviewer should inspect every output for factual fidelity, preserved uncertainty, "
            "severity, anatomy, numbers, negation, and newly introduced advice. The automated CSV is a queue, "
            "not a clinical assessment.",
        ]
    )
    (output_dir / "ERROR_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return errors


def write_human_evaluation_template(comparison: pd.DataFrame, output_path: Path) -> None:
    """Save a blinded-ready human rating sheet."""
    columns = {
        "example_id": comparison["example_id"],
        "medical_text": comparison["medical_text"],
        "baseline_output": comparison["baseline_output"],
        "finetuned_output": comparison["finetuned_output"],
    }
    rating_columns = (
        "accuracy_baseline_1_to_5", "accuracy_finetuned_1_to_5",
        "clarity_baseline_1_to_5", "clarity_finetuned_1_to_5",
        "uncertainty_preservation_baseline_1_to_5", "uncertainty_preservation_finetuned_1_to_5",
        "safety_baseline_1_to_5", "safety_finetuned_1_to_5",
        "preferred_output", "reviewer_comments",
    )
    for column in rating_columns:
        columns[column] = ""
    pd.DataFrame(columns).to_csv(output_path, index=False)


def write_placeholder_outputs(output_dir: Path = Path("data/outputs")) -> None:
    """Create honest placeholder analysis files before model training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    message = "Results will be populated after running the training and evaluation pipeline."
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps({"status": "not_run", "message": message}, indent=2), encoding="utf-8"
    )
    (output_dir / "ERROR_ANALYSIS.md").write_text(
        f"# Error Analysis\n\n{message}\n\nNo model outputs or metric values are fabricated.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("baseline", "train", "adapted", "evaluate"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--skip-bertscore", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if args.command == "baseline":
        run_baseline_inference(model_name=args.model_name)
    elif args.command == "train":
        train_model(model_name=args.model_name, adapter_dir=args.adapter_dir, use_4bit=args.use_4bit)
    elif args.command == "adapted":
        run_adapted_inference(model_name=args.model_name, adapter_dir=args.adapter_dir)
    else:
        evaluate_outputs(skip_bertscore=args.skip_bertscore)


if __name__ == "__main__":
    main()
