# Explainable Brain Tumor MRI Classification

This project classifies a brain MRI image as `glioma`, `meningioma`, `pituitary`, or `normal`.
Its novel component is an explainable prediction app that shows the predicted class, confidence,
class probabilities, and a Grad-CAM heatmap.

> **Not for clinical diagnosis.** This is an educational computer-vision project.

## Rubric Coverage

### Three Models

| Approach | Implementation | Location |
|---|---|---|
| Naive baseline | Most-common-class classifier | `src/brain_tumor_ml/training.py::fit_baseline` |
| Classical ML | Logistic regression with intensity, histogram, edge, and thumbnail features | `src/brain_tumor_ml/features.py`, `fit_classical` |
| Deep learning | Small convolutional neural network with augmentation | `src/brain_tumor_ml/models.py::SmallCNN`, `fit_deep` |

All models use the same patient-grouped test set. The CNN is the deployed model.

### Focused Experiment

The experiment compares the same CNN:

1. Without data augmentation.
2. With random flips, rotations, and small translations.

The split, architecture, seed, image size, epochs, and batch size remain fixed. Results include
accuracy, macro precision, macro recall, and macro F1.

Files:

- Code: `src/brain_tumor_ml/experiment.py`
- Table: `data/outputs/augmentation_experiment.csv`
- Explanation: `data/outputs/augmentation_experiment.json`
- Figure: `data/outputs/augmentation_experiment.png`

### Evaluation And Error Analysis

Training reports accuracy, precision, recall, F1, confusion matrices, and multiclass ROC-AUC.
Recall is emphasized because missed tumor classes can be especially harmful in medical imaging.

Outputs:

- `models/model_comparison.csv`
- `models/metrics.json`
- `data/outputs/confusion_matrices.png`
- `data/outputs/error_analysis/error_analysis.csv`
- `data/outputs/error_analysis/images/`

The error-analysis report selects up to five incorrect CNN predictions. For each case it records
the true label, predicted label, confidence, possible reason, suggested improvement, and an
annotated copy of the image. If the test set contains fewer than five errors, all available
errors are reported rather than inventing mistakes.

## Kaggle Dataset Preparation

The project supports:

- `masoudnickparvar/brain-tumor-mri-dataset`
- Dataset page: `https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset`

Install dependencies, then download and prepare it:

```bash
pip install -r requirements.txt
python main.py prepare-kaggle
```

`kagglehub` downloads the latest available version. The preparation command:

- preserves Kaggle's original `Testing` folder as the project test set;
- splits only Kaggle's `Training` folder into training and validation data;
- maps Kaggle's `notumor` folder to this project's `normal` label;
- creates the following reproducible structure:

```text
data/processed/kaggle/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── normal/
├── val/
│   └── ...
└── test/
    └── ...
```

Train all three required models and run the augmentation experiment:

```bash
python main.py pipeline \
  --data-dir data/processed/kaggle \
  --image-size 128 \
  --epochs 15
```

Do not pass the synthetic-data patient regex for this dataset. Its distributed filenames do not
provide reliable patient identifiers. Therefore, the code preserves the publisher's test split,
but it cannot independently verify patient-level separation. This limitation must be disclosed in
the report.

If you already downloaded the dataset manually:

```bash
python main.py prepare-kaggle --source-dir /path/to/downloaded/dataset
```

The pipeline then resizes images, converts them to grayscale, normalizes pixels, and applies
augmentation only to CNN training images.

## Setup In VS Code

Open the repository folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

If both `(.venv)` and `(base)` appear on macOS, deactivate Conda first:

```bash
conda deactivate
source .venv/bin/activate
```

## Quick Test With Synthetic Data

```bash
python main.py demo-data --samples-per-class 30

python main.py pipeline \
  --data-dir data/processed/demo \
  --patient-id-regex '(patient_\d+)' \
  --image-size 64 \
  --epochs 15
```

Synthetic data only verifies that the software works. Do not use its scores as medical results.
Very short CNN runs can collapse to one class. Check `models/model_comparison.csv` before opening
the app; the CNN should clearly outperform the 25% balanced-accuracy naive baseline.

## Run With Other MRI Data

Train and evaluate all three models:

```bash
python main.py train \
  --data-dir data/raw \
  --patient-id-regex '(patient_\d+)' \
  --epochs 15
```

Run the augmentation experiment:

```bash
python main.py experiment --epochs 10
```

If each filename is already one unique patient image, omit `--patient-id-regex`.

## Run The Interactive App

After training:

```bash
python main.py serve
```

Open `http://127.0.0.1:8000`. The app supports image upload, tumor-class prediction, confidence,
class probabilities, Grad-CAM visualization, and a clinical-use disclaimer.

## Public Deployment

Deploy the FastAPI application to a service such as Render, Railway, or Google Cloud Run.
The start command is:

```bash
uvicorn brain_tumor_ml.api:app --host 0.0.0.0 --port $PORT
```

Set `BRAIN_TUMOR_ARTIFACT_DIR=models` and ensure the trained files in `models/` are available to
the deployment. Hosting and uptime must be configured on the selected platform; repository code
cannot by itself guarantee that the public app remains live for one week.

## Tests

```bash
pytest -q
```

## Attribution

This project was developed with assistance from OpenAI ChatGPT for software engineering support, code generation, testing, and documentation. All generated content was reviewed, modified, and validated by the author.

Additional references:
- PyTorch Documentation: https://pytorch.org/docs/stable/
- scikit-learn Documentation: https://scikit-learn.org/stable/