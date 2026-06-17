"""Streamlit demo app for the MindSignal triage assistant."""

from __future__ import annotations

import streamlit as st

from mindsignal_utils import load_model_and_tokenizer, predict_text


st.set_page_config(page_title="MindSignal", page_icon="MS", layout="centered")

st.title("MindSignal: Mental Health Support Triage Assistant")
st.caption("Prototype classifier for short mental-health-related messages.")

st.warning(
    "Disclaimer: MindSignal is not a medical diagnosis tool, therapist, crisis line, "
    "or emergency service. If someone may be in immediate danger, contact local "
    "emergency services or a crisis hotline right away."
)


@st.cache_resource
def cached_model():
    """Load the model once so Streamlit interactions stay fast."""

    return load_model_and_tokenizer()


message = st.text_area(
    "User message",
    height=160,
    placeholder="Type a short message here...",
)

if st.button("Classify", type="primary"):
    if not message.strip():
        st.error("Please enter a message to classify.")
    else:
        try:
            tokenizer, model = cached_model()
            prediction = predict_text(message, tokenizer, model)
        except FileNotFoundError as error:
            st.error(str(error))
            st.stop()

        st.subheader("Prediction")
        st.metric("Label", prediction.label)
        st.metric("Confidence", f"{prediction.confidence:.2%}")

        if prediction.used_safety_override:
            st.info("Rule-based safety override was triggered by high-risk wording.")

        if prediction.label == "escalation_required":
            st.error(
                "Safety warning: this message may need urgent escalation. "
                "Encourage the person to contact emergency services, a trusted person, "
                "or a crisis support line immediately."
            )
