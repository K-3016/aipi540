# Report Outline

## Problem Statement

Describe the need for faster student support message routing and define category and urgency prediction tasks.

## Data Sources

Explain that the dataset is synthetic, generated for this project, and contains no real student records.

## Related Work

Summarize text classification, support-ticket triage, TF-IDF baselines, Logistic Regression, and transformer-based classifiers.

## Evaluation Strategy & Metrics

Describe train/validation/test splitting, accuracy, macro F1, weighted F1, per-class precision/recall/F1, and confusion matrices.

## Modeling Approach

Explain majority baseline, TF-IDF Logistic Regression, and optional DistilBERT fine-tuning.

## Data Processing Pipeline

Describe message generation, noise injection, stratified splitting, feature extraction, training, evaluation, and app inference.

## Hyperparameter Tuning Strategy

Discuss TF-IDF n-grams, maximum features, class weighting, Logistic Regression iterations, and optional transformer epoch/batch settings.

## Models Evaluated

List majority baseline, TF-IDF Logistic Regression, and optional DistilBERT category/urgency classifiers.

## Results

Reference `model_comparison.csv`, confusion matrices, and classification reports.

## Error Analysis

Reference `error_analysis.csv` and summarize short-message, ambiguous-language, and noisy-text failure modes.

## Experiment Write-Up

Describe the noisy-text robustness experiment and compare clean/noisy macro F1.

## Recommendations

Recommend the classical model for lightweight deployment with human review thresholds.

## Conclusions

Summarize routing benefits, limitations, and requirements before real deployment.

## Future Work

Discuss real labeled data, fairness checks, multilingual support, calibrated confidence, and workflow integration.

## Commercial Viability Statement

Assess value for university support operations, integration requirements, cost profile, and compliance constraints.

## Ethics Statement

Discuss privacy, bias, accessibility, student safety, synthetic-data limits, and human oversight.
