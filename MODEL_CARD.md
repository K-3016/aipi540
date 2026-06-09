# Model Card: Brain MRI Tumor Classifier

## Intended Use

Educational and research evaluation of four-class MRI image-classification workflows. The model
predicts `glioma`, `meningioma`, `pituitary`, or `normal`.

## Out-of-Scope Use

- Clinical diagnosis, triage, treatment, prognosis, or autonomous decision-making
- Use on modalities, anatomy, scanners, protocols, or populations absent from validation
- Uploading identifiable health information to an unsecured deployment
- Interpreting synthetic demo results as evidence of medical performance

## Models

The repository trains a naive prior classifier, a classical logistic-regression pipeline using
handcrafted image features, and a compact CNN. The CNN is deployed through FastAPI.

## Inputs and Outputs

Input: one raster image converted to grayscale and resized to the artifact's configured image
size. Output: one predicted class and probabilities for all four classes.

## Evaluation

All approaches use the same patient-grouped train/validation/test manifest. Results are written
to `metrics.json`, including macro and per-class metrics, multiclass discrimination and
calibration metrics, and a confusion matrix. The focused experiment compares CNN performance
with and without training-time augmentation. Grad-CAM overlays expose influential image regions
for qualitative review, and an error-analysis report records up to five incorrect predictions.

## Known Limitations

- A two-dimensional image may omit volumetric, sequence, and clinical context.
- Internal test performance does not establish transportability.
- Labels can encode site, scanner, annotation, or preprocessing artifacts.
- Softmax probabilities are not guaranteed to be clinically calibrated.
- Grad-CAM can be visually plausible even when a model relies on invalid shortcuts.
- No uncertainty interval, subgroup audit, external validation, or prospective study is included.

## Monitoring Recommendations

Monitor input validity, class and probability drift, scanner/site composition, missing sequences,
latency, failure rates, calibration, per-class sensitivity, and subgroup outcomes. Establish
human review and stop-use criteria before any real-world pilot.
