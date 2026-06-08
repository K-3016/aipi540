# Explainable Deep Learning for Brain Tumor MRI Classification

A reproducible computer-vision project that classifies brain MRI images as `glioma`,
`meningioma`, `pituitary`, or `normal`. It includes a naive baseline, classical machine
learning, deep learning, Grad-CAM explanations, confidence-based triage, robustness testing,
and a FastAPI web interface.

> **Research and education only.** This software is not a medical device and must not be used
> to diagnose, triage, or treat a patient.

## Repository Structure

```text
├── README.md
├── requirements.txt
├── Makefile
├── setup.py
├── main.py
├── scripts
│   ├── make_dataset.py
│   ├── build_features.py
│   └── model.py
├── models
│   └── .gitkeep
├── data
│   ├── raw
│   ├── processed
│   └── outputs
├── notebooks
├── src
│   └── brain_tumor_ml
├── tests
└── .gitignore
```

The required assignment-facing structure is at the repository root. Reusable implementation
code remains in `src/brain_tumor_ml/`, and automated tests remain in `tests/`.

## Requirement Map

| Requirement | Implementation | Location |
|---|---|---|
| Naive baseline | Prior classifier | `src/brain_tumor_ml/training.py` |
| Classical ML | Handcrafted features and logistic regression | `scripts/build_features.py`, `src/brain_tumor_ml/features.py` |
| Deep learning | Compact CNN or ResNet18 transfer learning | `scripts/model.py`, `src/brain_tumor_ml/models.py` |
| Focused experiment | Gaussian-noise robustness for all models | `src/brain_tumor_ml/experiment.py` |
| Explainability | Grad-CAM overlay | `src/brain_tumor_ml/explainability.py` |
| Deployment | Upload UI and JSON API | `main.py serve`, `src/brain_tumor_ml/api.py` |

All three approaches use the same untouched, patient-grouped test split.

## Setup

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/make_dataset.py
pytest -q
```

`setup.py` provides compatibility with standard editable installation:

```bash
pip install -e .
```

The Makefile combines installation and dataset-folder setup:

```bash
make setup
```

## Dataset

Place de-identified images into:

```text
data/raw/
├── glioma/
├── meningioma/
├── pituitary/
└── normal/
```

PNG, JPEG, BMP, and TIFF are supported. Folder names define labels. Review the source dataset's
license, label definitions, and privacy conditions before use.

Multiple slices from one patient must not cross train, validation, and test splits. Pass a
regular expression that extracts patient ID from each filename:

```bash
python3 main.py train \
  --data-dir data/raw \
  --model-dir models \
  --patient-id-regex '(patient_\d+)'
```

Without `--patient-id-regex`, each image stem is treated as a separate patient. For a real MRI
dataset with multiple images per person, always provide patient grouping.

The repository does not automatically redistribute a medical dataset. Obtain an appropriately
licensed, de-identified dataset and document its source and license. `scripts/make_dataset.py`
creates the required class folders.

## Run The Pipeline

For a software-only smoke test, generate artificial images:

```bash
python3 main.py demo-data --samples-per-class 18
python3 main.py pipeline \
  --data-dir data/processed/demo \
  --model-dir models \
  --report-dir data/outputs \
  --patient-id-regex '(patient_\d+)' \
  --image-size 64 \
  --epochs 3
```

Synthetic metrics only verify that the software works. They have no clinical meaning.

## Deep Learning Options

The CPU-friendly baseline CNN is the default:

```bash
python3 main.py train --data-dir data/raw --model-dir models
```

Use ResNet18 transfer learning when internet access is available for the initial ImageNet weight
download:

```bash
python3 main.py train \
  --data-dir data/raw \
  --model-dir models \
  --architecture resnet18 \
  --pretrained
```

Omit `--pretrained` to train the ResNet18 architecture from random initialization. To compare
architectures fairly, use the same manifest or seed, training budget, and evaluation set.

## Evaluation

Training writes these files to `models/`:

- `metrics.json`: test results for all three approaches
- `manifest.csv`: exact patient-grouped split
- `training_history.json`: deep-model train and validation losses
- `baseline.joblib`, `classical.joblib`, `deep_model.pt`: model artifacts
- `metadata.json`: class order, architecture, image size, seed, and split counts

Metrics include accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, one-vs-rest
macro ROC-AUC, log loss, multiclass Brier score, per-class sensitivity/specificity, and a 4x4
confusion matrix. Each model also receives a confidence-triage table showing coverage and
retained-case accuracy at thresholds from 0.5 to 0.9.

The focused experiment writes `data/outputs/noise_robustness.json` and
`data/outputs/noise_robustness.png`.

Individual pipeline stages are also available:

```bash
python3 scripts/build_features.py --patient-id-regex '(patient_\d+)'
python3 scripts/model.py train --patient-id-regex '(patient_\d+)'
python3 main.py experiment
```

## Deployment

```bash
python3 main.py serve --model-dir models
```

Open `http://127.0.0.1:8000`, or call:

```bash
curl -F "file=@example.png" http://127.0.0.1:8000/predict
```

The response contains the predicted class, confidence, all four probabilities, a human-review
recommendation, architecture, and a medical-use disclaimer. `POST /explain` additionally returns
a base64-encoded Grad-CAM overlay. The browser UI uses this endpoint automatically. Grad-CAM is
an attention visualization, not proof of clinically valid reasoning.

## Before Reporting Real-World Performance

1. Document the dataset, label reference standard, exclusions, and class distribution.
2. Audit duplicate images and patient leakage.
3. Reserve an external institution or later time period for testing.
4. Add bootstrap confidence intervals and subgroup analyses.
5. Evaluate calibration and clinically relevant operating behavior.
6. Test scanner, sequence, compression, and acquisition-quality shifts.
7. Complete privacy, security, bias, regulatory, and prospective-validation reviews.

## Attribution

This project was developed with assistance from OpenAI ChatGPT for software engineering support, code generation, testing, and documentation. All generated content was reviewed, modified, and validated by the author.

Additional references:
- PyTorch Documentation: https://pytorch.org/docs/stable/
- scikit-learn Documentation: https://scikit-learn.org/stable/