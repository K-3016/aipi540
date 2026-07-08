# Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""Streamlit investor-demo app for campus support triage.

Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent

for import_path in (SRC_DIR, ROOT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import streamlit as st

from campus_triage.config import CATEGORY_LABELS, URGENCY_LABELS
from campus_triage.predict import EXAMPLE_MESSAGES, load_deployed_model, model_available, model_search_diagnostics, predict_message


CUSTOM_CSS = """
<style>
    :root {
        --ink: #111827;
        --muted: #64748b;
        --line: #d9e2ec;
        --paper: #ffffff;
        --surface: #f5f7fb;
        --teal: #0f766e;
        --navy: #172554;
        --amber: #b45309;
        --rose: #be123c;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(15, 118, 110, .12), transparent 30rem),
            linear-gradient(180deg, #f8fafc 0%, #eef3f8 100%);
        color: var(--ink);
    }
    .block-container { padding-top: 1.6rem; max-width: 1240px; }
    h1, h2, h3, p { letter-spacing: 0; }
    div[data-testid="stToolbar"] { display: none; }
    .hero {
        padding: 1.65rem 1.85rem;
        border-radius: 8px;
        background: linear-gradient(135deg, #0f172a 0%, #155e75 55%, #0f766e 100%);
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 18px 45px rgba(15, 23, 42, .20);
    }
    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        padding: .25rem .55rem;
        border: 1px solid rgba(255, 255, 255, .28);
        border-radius: 999px;
        color: #ccfbf1;
        font-size: .78rem;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: .75rem;
    }
    .hero h1 { font-size: clamp(2rem, 4vw, 3.35rem); line-height: 1.02; margin: 0 0 .55rem 0; }
    .hero p { font-size: 1.05rem; max-width: 780px; margin: 0; color: #dbeafe; }
    .status-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: .7rem;
        margin: .85rem 0 1.1rem 0;
    }
    .status-item {
        background: rgba(255, 255, 255, .88);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: .75rem .9rem;
        min-height: 78px;
    }
    .status-label { color: var(--muted); font-size: .75rem; text-transform: uppercase; font-weight: 800; }
    .status-value { color: var(--ink); font-size: 1rem; font-weight: 850; margin-top: .18rem; }
    .panel {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1.15rem;
        box-shadow: 0 16px 34px rgba(15, 23, 42, .08);
    }
    .panel-title { margin: 0 0 .25rem 0; color: var(--ink); font-size: 1.1rem; font-weight: 850; }
    .panel-subtitle { color: var(--muted); font-size: .9rem; margin: 0 0 .9rem 0; }
    .result-card {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: .95rem 1rem;
        min-height: 112px;
    }
    .result-label { color: var(--muted); font-size: .74rem; text-transform: uppercase; font-weight: 850; }
    .result-value { color: var(--ink); font-size: 1.48rem; line-height: 1.1; font-weight: 900; margin-top: .3rem; }
    .pill-row { display: flex; gap: .45rem; flex-wrap: wrap; margin-top: .65rem; }
    .pill {
        display: inline-flex;
        border-radius: 999px;
        padding: .22rem .55rem;
        font-size: .76rem;
        font-weight: 800;
        background: #e0f2fe;
        color: #075985;
    }
    .urgency-high { background: #fff1f2; color: var(--rose); border: 1px solid #fecdd3; }
    .urgency-medium { background: #fffbeb; color: var(--amber); border: 1px solid #fde68a; }
    .urgency-low { background: #ecfdf5; color: #047857; border: 1px solid #bbf7d0; }
    .recommendation {
        background: linear-gradient(90deg, #ecfdf5 0%, #eff6ff 100%);
        border: 1px solid #99f6e4;
        border-left: 6px solid var(--teal);
        border-radius: 8px;
        padding: 1rem 1.1rem;
        color: #064e3b;
        margin-top: 1rem;
    }
    .recommendation strong { color: #0f172a; }
    .explain {
        background: #f8fafc;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: .9rem 1rem;
        color: #334155;
        margin-top: .85rem;
    }
    .score-row {
        display: grid;
        grid-template-columns: minmax(120px, 190px) 1fr 46px;
        gap: .7rem;
        align-items: center;
        margin: .44rem 0;
    }
    .score-name { font-weight: 750; color: #334155; font-size: .88rem; }
    .score-track { height: .65rem; background: #e5e7eb; border-radius: 999px; overflow: hidden; }
    .score-fill { height: 100%; background: linear-gradient(90deg, #0f766e, #2563eb); border-radius: 999px; }
    .score-value { text-align: right; color: #475569; font-variant-numeric: tabular-nums; font-weight: 800; }
    .footer-note {
        color: #475569;
        font-size: .86rem;
        border-top: 1px solid var(--line);
        margin-top: 1rem;
        padding-top: .8rem;
    }
    textarea, .stTextArea textarea {
        color: #111827 !important;
        background-color: #ffffff !important;
        border: 1px solid #94a3b8 !important;
        border-radius: 8px !important;
        caret-color: #0f766e !important;
        font-size: 1rem !important;
        line-height: 1.45 !important;
    }
    .stTextArea textarea:focus {
        border-color: #0f766e !important;
        box-shadow: 0 0 0 3px rgba(15, 118, 110, .16) !important;
    }
    div[data-testid="stSelectbox"] div { color: #111827; }
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 850 !important;
        border: 1px solid #0f766e !important;
        min-height: 2.75rem;
    }
    .stButton > button[kind="primary"] {
        background: #0f766e !important;
        color: white !important;
    }
    @media (max-width: 800px) {
        .status-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .hero { padding: 1.25rem; }
        .score-row { grid-template-columns: 1fr; gap: .25rem; }
        .score-value { text-align: left; }
    }
</style>
"""


def display_label(label: str) -> str:
    """Convert model labels into polished display text."""

    return label.replace("_", " ").title()


def confidence_band(score: float) -> str:
    """Convert confidence into a product-facing review band."""

    if score >= 0.75:
        return "Auto-route candidate"
    if score >= 0.55:
        return "Route with review"
    return "Manual review recommended"


def urgency_class(urgency: str) -> str:
    """Return the CSS class for an urgency pill."""

    return {
        "high": "urgency-high",
        "medium": "urgency-medium",
        "low": "urgency-low",
    }.get(urgency, "urgency-medium")


def render_status_strip() -> None:
    """Render investor-demo operating metrics."""

    st.markdown(
        """
        <div class="status-strip">
            <div class="status-item"><div class="status-label">Model Mode</div><div class="status-value">Inference Only</div></div>
            <div class="status-item"><div class="status-label">Routing Targets</div><div class="status-value">7 Queues</div></div>
            <div class="status-item"><div class="status-label">Urgency Tiers</div><div class="status-value">Low / Medium / High</div></div>
            <div class="status-item"><div class="status-label">Default Model</div><div class="status-value">TF-IDF Logistic Regression</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_score_table(scores: dict[str, float], labels: list[str]) -> None:
    """Render confidence scores as a compact leaderboard."""

    ordered_labels = sorted(labels, key=lambda label: scores.get(label, 0.0), reverse=True)
    for label in ordered_labels:
        score = max(0.0, min(scores.get(label, 0.0), 1.0))
        st.markdown(
            f"""
            <div class="score-row">
                <div class="score-name">{html.escape(display_label(label))}</div>
                <div class="score-track"><div class="score-fill" style="width: {score * 100:.1f}%;"></div></div>
                <div class="score-value">{score:.0%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_example_buttons() -> None:
    """Render example messages as quick-fill controls."""

    st.caption("Demo-ready examples")
    for index, example in enumerate(EXAMPLE_MESSAGES, start=1):
        label = f"Example {index}: {example[:54]}{'...' if len(example) > 54 else ''}"
        if st.button(label, key=f"example_{index}", use_container_width=True):
            st.session_state["message_text"] = example


def render_empty_decision_panel() -> None:
    """Render the pre-analysis decision panel."""

    st.markdown('<div class="panel"><p class="panel-title">Triage Decision</p><p class="panel-subtitle">Run an analysis to generate routing output, confidence, and next action.</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="result-card">
            <div class="result-label">Current Status</div>
            <div class="result-value">Ready</div>
            <div class="pill-row"><span class="pill">Model loaded</span><span class="pill">Synthetic POC</span></div>
        </div>
        <div class="footer-note">This proof of concept supports triage decisions. It does not replace human review, crisis response workflows, or official university policy.</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction(prediction: dict[str, object]) -> None:
    """Render prediction cards, recommendation, and confidence evidence."""

    category = str(prediction["category"])
    urgency = str(prediction["urgency"])
    category_confidence = float(prediction["category_confidence"])
    urgency_confidence = float(prediction["urgency_confidence"])
    band = confidence_band(category_confidence)

    st.markdown('<div class="panel"><p class="panel-title">Triage Decision</p><p class="panel-subtitle">Operational output for the support intake queue.</p>', unsafe_allow_html=True)
    result_columns = st.columns(3)
    with result_columns[0]:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Predicted Queue</div>
                <div class="result-value">{html.escape(display_label(category))}</div>
                <div class="pill-row"><span class="pill">{category_confidence:.0%} confidence</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with result_columns[1]:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Urgency Tier</div>
                <div class="result-value">{html.escape(display_label(urgency))}</div>
                <div class="pill-row"><span class="pill {urgency_class(urgency)}">{urgency_confidence:.0%} confidence</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with result_columns[2]:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Review Posture</div>
                <div class="result-value">{html.escape(band)}</div>
                <div class="pill-row"><span class="pill">Human-in-loop</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="recommendation"><strong>Recommended next action:</strong> {html.escape(str(prediction["routing_recommendation"]))}</div>
        <div class="explain"><strong>Why this route:</strong> {html.escape(str(prediction["explanation"]))}</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    score_columns = st.columns(2)
    with score_columns[0]:
        st.markdown('<div class="panel"><p class="panel-title">Category Confidence</p><p class="panel-subtitle">Ranked routing probabilities.</p>', unsafe_allow_html=True)
        render_score_table(prediction["category_scores"], CATEGORY_LABELS)  # type: ignore[arg-type]
        st.markdown("</div>", unsafe_allow_html=True)
    with score_columns[1]:
        st.markdown('<div class="panel"><p class="panel-title">Urgency Confidence</p><p class="panel-subtitle">Ranked urgency probabilities.</p>', unsafe_allow_html=True)
        render_score_table(prediction["urgency_scores"], URGENCY_LABELS)  # type: ignore[arg-type]
        st.markdown("</div>", unsafe_allow_html=True)


def run_app() -> None:
    """Run the Streamlit application."""

    st.set_page_config(page_title="Campus Triage Assistant", page_icon="CS", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <section class="hero">
            <h1>Campus Support Message Triage Assistant</h1>
            <p>AI-assisted intake for university support teams: classify incoming student messages, estimate urgency, and produce routing guidance with confidence evidence.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    render_status_strip()

    if not model_available():
        st.error("No trained model artifact was found for inference.")
        st.caption("The app checked these local and Hugging Face deployment paths:")
        st.code(model_search_diagnostics())
        st.stop()

    model = load_deployed_model()
    if "message_text" not in st.session_state:
        st.session_state["message_text"] = EXAMPLE_MESSAGES[0]

    intake_column, decision_column = st.columns([1.05, 1.15], gap="large")
    with intake_column:
        st.markdown('<div class="panel"><p class="panel-title">Message Intake</p><p class="panel-subtitle">Paste a support request or load a demo scenario.</p>', unsafe_allow_html=True)
        render_example_buttons()
        message_text = st.text_area(
            "Student message",
            key="message_text",
            height=210,
            label_visibility="collapsed",
        )
        submitted = st.button("Analyze and Route Message", type="primary", use_container_width=True)
        st.markdown(
            """
            <div class="footer-note">Proof-of-concept only. High-risk health, safety, and crisis messages require established human escalation procedures.</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with decision_column:
        if submitted and message_text.strip():
            prediction = predict_message(message_text, model=model)
            render_prediction(prediction)
        elif submitted:
            st.warning("Paste a student message before analyzing.")
            render_empty_decision_panel()
        else:
            render_empty_decision_panel()

    st.markdown(
        """
        <div class="footer-note">Model note: deployed inference uses a lightweight TF-IDF Logistic Regression model trained on synthetic course data. Use confidence thresholds and human review before any real operational deployment.</div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    run_app()
