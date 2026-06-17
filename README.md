# MindSignal: Mental Health Support Triage Assistant

MindSignal is a hackathon prototype that classifies short mental-health-related messages into three triage labels:

- `informational`
- `emotional_support_needed`
- `escalation_required`

## Problem Statement

Support teams, community moderators, and wellness platforms often receive short messages that vary widely in urgency. MindSignal explores whether a lightweight NLP assistant can help triage these messages into informational requests, emotional support needs, and possible escalation cases.

This project is a prototype only. It is not a medical diagnosis tool, therapist, crisis service, or emergency system.

## Model Used

The project uses the pre-trained Hugging Face model `distilbert-base-uncased`.

DistilBERT is a smaller and faster version of BERT. It already understands a broad amount of English language structure from pre-training, which makes it useful for transfer learning on a small hackathon dataset.

## Transfer Learning Approach

`train.py` loads the pre-trained DistilBERT weights and adds a 3-class sequence classification head:

- class 0: `informational`
- class 1: `emotional_support_needed`
- class 2: `escalation_required`

The model is then fine-tuned on rows where `split == "train"` in:

```text
data/mental_health_triage_synthetic_dataset.csv
```

The fine-tuned model is saved to:

```text
models/mindsignal-distilbert/
```

## Data Augmentation

The training pipeline keeps the original examples and adds lightweight augmented copies using:

- random lowercase conversion
- small typo/noise injection
- emoji insertion
- slang phrase insertion
- abbreviation examples such as `rn`, `ngl`, and `idk`

These augmentations are intentionally simple and are used only on the training split.

## Safety Override

Before model prediction, MindSignal checks for high-risk phrases such as:

- `kill myself`
- `end my life`
- `not worth living`
- `hurt myself`
- `suicide`
- `not safe`
- `better off without me`
- `can't keep myself safe`

If one of these phrases is found, the prediction is immediately set to:

```text
escalation_required
```

This rule-based override is used in both `evaluate.py` and `app.py`.

## Evaluation Metrics

`evaluate.py` evaluates the test split and stress-test split separately.

It reports:

- accuracy
- macro F1
- per-class precision, recall, and F1
- `escalation_required` recall
- confusion matrix
- stress-test accuracy

Outputs are saved to:

```text
results/evaluation_report.txt
results/confusion_matrix.png
```

## Preliminary Results

After training, run evaluation to generate preliminary results:

```bash
python evaluate.py
```

The generated report will contain the current model's metrics on the `test` and `stress_test` splits. Because this repository includes a very small synthetic starter dataset, results are useful only as a smoke test. Replace the CSV with a larger reviewed dataset for meaningful model performance.

## Project Structure

```text
.
+-- app.py
+-- evaluate.py
+-- train.py
+-- mindsignal_utils.py
+-- requirements.txt
+-- README.md
+-- data/
|   +-- mental_health_triage_synthetic_dataset.csv
+-- models/
|   +-- mindsignal-distilbert/
+-- results/
    +-- evaluation_report.txt
    +-- confusion_matrix.png
```

## Setup

Use Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Training

```bash
python train.py
```

This fine-tunes DistilBERT and saves the trained model in `models/mindsignal-distilbert/`.

## Run Evaluation

```bash
python evaluate.py
```

This writes `results/evaluation_report.txt` and `results/confusion_matrix.png`.

## Run the Streamlit App

```bash
streamlit run app.py
```

The app provides:

- a text area for a user message
- a classify button
- predicted label
- confidence score
- safety warning when `escalation_required` is predicted
- a clear medical and emergency-service disclaimer

## Limitations

- The included dataset is synthetic and small.
- The classifier should not be used as a standalone safety system.
- High-risk language can be indirect, sarcastic, multilingual, or misspelled.
- False negatives are especially serious for escalation cases.
- Human review and professional crisis workflows are required for real deployments.
- More diverse data, expert labeling, calibration, fairness testing, and red-team evaluation are needed before real-world use.
## Attribution

This project was developed with assistance from OpenAI ChatGPT for software engineering support, code generation, testing, and documentation. All generated content was reviewed, modified, and validated by the author.

Additional references:
- PyTorch Documentation: https://pytorch.org/docs/stable/
- scikit-learn Documentation: https://scikit-learn.org/stable/
