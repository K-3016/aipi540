"""MedExplain command-line interface and Streamlit demonstration."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.build_features import build_features
from scripts.make_dataset import build_dataset
from scripts.model import (
    DEFAULT_ADAPTER,
    DEFAULT_MODEL,
    DISCLAIMER,
    detect_device,
    evaluate_outputs,
    generate_text,
    load_adapted_model,
    load_base_model,
    load_tokenizer,
    readability_metrics,
    run_adapted_inference,
    run_baseline_inference,
    safety_heuristics,
    train_model,
    validate_model_path,
    write_placeholder_outputs,
)

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA = PROJECT_ROOT / "data/raw/medical_rewrite_dataset.jsonl"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_DIR = PROJECT_ROOT / "data/outputs"
SAVED_COMPARISONS = OUTPUT_DIR / "comparison_results.csv"


def prepare_data(num_examples: int = 500, seed: int = 42, source: Path | None = None) -> None:
    """Generate/validate raw records and construct deterministic data splits."""
    build_dataset(RAW_DATA, num_examples, seed, source=source)
    build_features(RAW_DATA, PROCESSED_DIR, seed)
    write_placeholder_outputs(OUTPUT_DIR)


def train_workflow(
    model_name: str,
    adapter_dir: Path,
    use_4bit: bool,
    epochs: float,
    seed: int,
) -> None:
    """Save baseline predictions, then train and save a LoRA adapter."""
    require_prepared_data()
    run_baseline_inference(model_name=model_name)
    train_model(
        model_name=model_name,
        adapter_dir=adapter_dir,
        use_4bit=use_4bit,
        epochs=epochs,
        seed=seed,
    )


def evaluation_workflow(
    model_name: str,
    adapter_dir: Path,
    skip_bertscore: bool,
) -> dict[str, Any]:
    """Generate adapted predictions and evaluate paired model outputs."""
    require_prepared_data()
    baseline_path = OUTPUT_DIR / "baseline_predictions.csv"
    if not baseline_path.exists():
        LOGGER.info("Baseline predictions are missing; generating them now.")
        run_baseline_inference(model_name=model_name)
    run_adapted_inference(model_name=model_name, adapter_dir=adapter_dir)
    return evaluate_outputs(skip_bertscore=skip_bertscore)


def require_prepared_data() -> None:
    """Fail with an actionable message if preparation has not run."""
    required = [PROCESSED_DIR / f"{split}.jsonl" for split in ("train", "validation", "test")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Prepared data are missing. Run `python main.py prepare-data` first. "
            f"Missing: {', '.join(missing)}"
        )


def predict_pair(text: str, model_name: str, adapter_dir: Path) -> dict[str, Any]:
    """Generate base and adapted rewrites for one supplied medical statement."""
    clean = text.strip()
    if not clean:
        raise ValueError("Prediction text must not be empty.")
    tokenizer = load_tokenizer(model_name)
    base_model = load_base_model(model_name)
    baseline = generate_text(base_model, tokenizer, clean)
    adapted_model, adapted_tokenizer = load_adapted_model(model_name, adapter_dir)
    finetuned = generate_text(adapted_model, adapted_tokenizer, clean)
    return {
        "original_medical_text": clean,
        "base_model_output": baseline,
        "fine_tuned_model_output": finetuned,
        "baseline_safety_flags": safety_heuristics(clean, baseline),
        "finetuned_safety_flags": safety_heuristics(clean, finetuned),
        "disclaimer": DISCLAIMER,
    }


def launch_streamlit() -> None:
    """Launch this file through Streamlit using the current Python environment."""
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())], check=True)


def render_streamlit_app() -> None:
    """Render the interactive before/after demonstration."""
    import streamlit as st

    st.set_page_config(page_title="MedExplain", page_icon="🩺", layout="wide")
    st.title("MedExplain: Patient-Friendly Medical Rewrite Tool")
    st.warning(DISCLAIMER)
    st.caption(
        "Hackathon prototype · rewriting only · not clinically validated · "
        f"base model: {DEFAULT_MODEL}"
    )

    mode = st.radio(
        "Demonstration mode",
        ["Saved held-out comparisons", "Live inference (local/GPU only)"],
        horizontal=True,
        help="Saved mode is the reliable option on memory-limited Streamlit Community Cloud.",
    )

    if mode == "Saved held-out comparisons":
        _render_saved_comparison(st)
        return

    st.error(
        "Live inference loads the 1.5B base model and can exceed Streamlit Community "
        "Cloud's memory limit. Use this mode locally or in a GPU environment."
    )
    medical_text = st.text_area(
        "Paste a medical statement or report excerpt",
        height=180,
        placeholder="Paste only non-identifiable medical text.",
    )

    @st.cache_resource(show_spinner=False)
    def cached_adapted() -> tuple[Any, Any]:
        use_4bit = detect_device() == "cpu" and platform.system() == "Linux"
        return load_adapted_model(DEFAULT_MODEL, DEFAULT_ADAPTER, use_4bit=use_4bit)

    if st.button("Rewrite and compare", type="primary", width="stretch"):
        if not medical_text.strip():
            st.error("Enter a medical statement before requesting a rewrite.")
            return
        try:
            validate_model_path(DEFAULT_ADAPTER)
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
            st.info("The app intentionally does not show invented fine-tuned results.")
            return
        device = detect_device()
        if device == "cpu":
            st.warning(
                "Running on CPU with memory-saving 4-bit loading on Linux. The first "
                "request can take several minutes; later requests reuse the cached model."
            )
        progress = st.status("Preparing the comparison…", expanded=True)
        try:
            progress.write("Loading the base model and LoRA adapter (one-time startup)…")
            adapted_model, adapted_tokenizer = cached_adapted()
            progress.write("Generating the base-model rewrite…")
            with adapted_model.disable_adapter():
                baseline = generate_text(
                    adapted_model,
                    adapted_tokenizer,
                    medical_text,
                    max_new_tokens=80,
                )
            progress.write("Generating the LoRA-adapted rewrite…")
            finetuned = generate_text(
                adapted_model,
                adapted_tokenizer,
                medical_text,
                max_new_tokens=80,
            )
            progress.update(label="Comparison ready", state="complete", expanded=False)
        except Exception as exc:
            LOGGER.exception("Inference failed")
            progress.update(label="Inference failed", state="error", expanded=True)
            st.error(f"Inference failed: {exc}")
            return

        st.subheader("Original medical text")
        st.write(medical_text)
        left, right = st.columns(2)
        with left:
            st.subheader("Base-model rewrite")
            st.write(baseline)
        with right:
            st.subheader("Fine-tuned-model rewrite")
            st.write(finetuned)

        rows = []
        for label, value in (
            ("Original", medical_text),
            ("Base model", baseline),
            ("Fine-tuned model", finetuned),
        ):
            metrics = readability_metrics(value)
            rows.append(
                {
                    "Text": label,
                    "Reading ease": round(metrics["flesch_reading_ease"], 2),
                    "Grade level": round(metrics["flesch_kincaid_grade"], 2),
                    "Words": int(metrics["word_count"]),
                    "Sentences": int(metrics["sentence_count"]),
                }
            )
        st.subheader("Readability comparison")
        st.dataframe(rows, width="stretch", hide_index=True)

        st.subheader("Safety heuristic flags")
        flag_left, flag_right = st.columns(2)
        with flag_left:
            st.write("Base model")
            st.json(safety_heuristics(medical_text, baseline))
        with flag_right:
            st.write("Fine-tuned model")
            st.json(safety_heuristics(medical_text, finetuned))
        st.caption("These rules are screening heuristics only; they cannot establish medical correctness or safety.")


def _clean_saved_generation(text: str) -> str:
    """Trim prompt-role spillover while preserving the generated first response."""
    clean = str(text).strip()
    for marker in ("Human:", "### Explanation", "### Response"):
        if marker in clean:
            clean = clean.split(marker, maxsplit=1)[0].strip()
    return clean


def _render_comparison(st: Any, medical_text: str, baseline: str, finetuned: str) -> None:
    """Display a paired rewrite, readability metrics, and heuristic safety flags."""
    st.subheader("Original medical text")
    st.write(medical_text)
    left, right = st.columns(2)
    with left:
        st.subheader("Base-model rewrite")
        st.write(baseline)
    with right:
        st.subheader("LoRA-adapted rewrite")
        st.write(finetuned)

    rows = []
    for label, value in (
        ("Original", medical_text),
        ("Base model", baseline),
        ("LoRA-adapted model", finetuned),
    ):
        metrics = readability_metrics(value)
        rows.append(
            {
                "Text": label,
                "Reading ease": round(metrics["flesch_reading_ease"], 2),
                "Grade level": round(metrics["flesch_kincaid_grade"], 2),
                "Words": int(metrics["word_count"]),
                "Sentences": int(metrics["sentence_count"]),
            }
        )
    st.subheader("Readability comparison")
    st.dataframe(rows, width="stretch", hide_index=True)

    st.subheader("Safety heuristic flags")
    flag_left, flag_right = st.columns(2)
    with flag_left:
        st.write("Base model")
        st.json(safety_heuristics(medical_text, baseline))
    with flag_right:
        st.write("LoRA-adapted model")
        st.json(safety_heuristics(medical_text, finetuned))
    st.caption(
        "These rules are screening heuristics only; they cannot establish medical "
        "correctness or safety."
    )


def _render_saved_comparison(st: Any) -> None:
    """Render genuine predictions generated during the completed evaluation run."""
    import pandas as pd

    st.info(
        "Cloud-safe demo: these are previously generated outputs from the pretrained "
        "base model and the LoRA-adapted model on the same held-out test examples. "
        "This view does not run live inference."
    )
    if not SAVED_COMPARISONS.exists():
        st.error(
            "Saved comparisons are unavailable. Run `python main.py evaluate` and "
            "include data/outputs/comparison_results.csv in the deployment."
        )
        return

    @st.cache_data
    def load_saved_comparisons(path: str) -> Any:
        frame = pd.read_csv(path)
        required = {
            "example_id",
            "medical_text",
            "reference_output",
            "baseline_output",
            "finetuned_output",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Saved comparison file is missing columns: {sorted(missing)}")
        return frame

    try:
        comparisons = load_saved_comparisons(str(SAVED_COMPARISONS))
    except (OSError, ValueError) as exc:
        st.error(f"Could not load saved comparisons: {exc}")
        return

    labels = {
        index: f"{row.example_id} — {str(row.medical_text)[:95]}"
        for index, row in comparisons.iterrows()
    }
    selected_index = st.selectbox(
        "Choose a held-out test example",
        options=list(labels),
        format_func=labels.get,
    )
    selected = comparisons.loc[selected_index]
    baseline = _clean_saved_generation(selected["baseline_output"])
    finetuned = _clean_saved_generation(selected["finetuned_output"])

    _render_comparison(st, str(selected["medical_text"]), baseline, finetuned)
    with st.expander("Reference patient-friendly rewrite"):
        st.write(selected["reference_output"])
    st.caption(
        "Outputs are authentic saved model generations. Prompt-role spillover after "
        "the first response is trimmed for display; no medical content is rewritten."
    )


def _is_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except (ImportError, RuntimeError):
        return False


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(description="MedExplain pipeline and demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Generate/validate and split data")
    prepare.add_argument("--num-examples", type=int, default=500)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--source", type=Path)

    train = subparsers.add_parser("train", help="Run baseline inference and LoRA training")
    train.add_argument("--model-name", default=DEFAULT_MODEL)
    train.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    train.add_argument("--use-4bit", action="store_true")
    train.add_argument("--epochs", type=float, default=2.0)
    train.add_argument("--seed", type=int, default=42)

    evaluate = subparsers.add_parser("evaluate", help="Run adapted inference and paired evaluation")
    evaluate.add_argument("--model-name", default=DEFAULT_MODEL)
    evaluate.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    evaluate.add_argument("--skip-bertscore", action="store_true")

    predict = subparsers.add_parser("predict", help="Compare base and adapted output for one text")
    predict.add_argument("--text", required=True)
    predict.add_argument("--model-name", default=DEFAULT_MODEL)
    predict.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)

    subparsers.add_parser("app", help="Launch the Streamlit demo")

    pipeline = subparsers.add_parser("pipeline", help="Run preparation, training, and evaluation")
    pipeline.add_argument("--num-examples", type=int, default=500)
    pipeline.add_argument("--seed", type=int, default=42)
    pipeline.add_argument("--model-name", default=DEFAULT_MODEL)
    pipeline.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    pipeline.add_argument("--use-4bit", action="store_true")
    pipeline.add_argument("--epochs", type=float, default=2.0)
    pipeline.add_argument("--skip-bertscore", action="store_true")
    return parser


def cli() -> None:
    """Dispatch command-line workflows."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    if args.command == "prepare-data":
        prepare_data(args.num_examples, args.seed, args.source)
    elif args.command == "train":
        train_workflow(args.model_name, args.adapter_dir, args.use_4bit, args.epochs, args.seed)
    elif args.command == "evaluate":
        summary = evaluation_workflow(args.model_name, args.adapter_dir, args.skip_bertscore)
        print(json.dumps(summary, indent=2))
    elif args.command == "predict":
        print(json.dumps(predict_pair(args.text, args.model_name, args.adapter_dir), indent=2))
    elif args.command == "app":
        launch_streamlit()
    elif args.command == "pipeline":
        prepare_data(args.num_examples, args.seed)
        train_workflow(args.model_name, args.adapter_dir, args.use_4bit, args.epochs, args.seed)
        summary = evaluation_workflow(args.model_name, args.adapter_dir, args.skip_bertscore)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if _is_streamlit_runtime():
        render_streamlit_app()
    else:
        cli()
