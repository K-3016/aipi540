# MedExplain Rubric Coverage Audit

Audit date: 2026-07-31

Rubric reviewed: `Module 3 Hackathon Rubric Kailong.pdf`

Status key: COVERED, PARTIAL, MISSING, or MISMATCH.

## Executive summary

MedExplain covers the core generative-model engineering requirement: it adapts `Qwen/Qwen2.5-1.5B-Instruct` with LoRA/PEFT, runs the trained adapter, and produces a paired held-out before/after comparison with evaluation and medical-safety reflection. The local repository tests pass (`19 passed`).

The project is not yet submission-complete under this PDF because it lacks public links, a recorded pitch, a deployed application, and Git branch/pull-request history. The PDF also contains recommendation-system requirements that conflict with MedExplain's explicitly scoped generative rewriting task; this should be clarified with the instructor rather than silently reframing MedExplain as a recommender.

## Core project requirements

| Rubric criterion | Status | Evidence or action |
|---|---|---|
| Addresses a recommendation problem where stakes go beyond convenience | MISMATCH | MedExplain is a high-stakes medical communication rewrite tool, not a recommendation system. Its specification explicitly prohibits diagnosis and treatment recommendations. Confirm whether this line was retained from a recommendation-system rubric by mistake. |
| Incorporates a real-world constraint such as cold start, fairness, diversity, or limited data | COVERED | Uses only 500 explicitly synthetic examples and discusses limited-data, bias, accessibility, privacy, and synthetic-data limitations in `README.md`. |
| Demonstrates how relevance is balanced with responsible recommendations | PARTIAL / MISMATCH | Paired evaluation measures rewrite relevance/fidelity against medical-safety heuristics. It does not evaluate recommendations because the system intentionally does not make them. Reframe verbally as usefulness versus faithful and responsible explanation only if the instructor confirms this interpretation. |

## Required submission links

| Rubric criterion | Status | Required action |
|---|---|---|
| Viewable short-pitch link submitted | MISSING | Record and upload a pitch of five minutes or less. |
| Pitch link accessible | MISSING | Test in a logged-out/private browser window. |
| GitHub repository link submitted | MISSING | Initialize/publish the repository and add the URL to the submission. |
| GitHub repository contains full codebase | PARTIAL | Full code exists locally. Verify it is pushed, including README, scripts, tests, and dependency files. Adapter and generated outputs are currently ignored by Git. |
| Deployed application link submitted | MISSING | Deploy the Streamlit application and submit its URL. |
| Deployed application link accessible | MISSING | Test from a logged-out/private browser and another device if possible. |

## Pitch

| Rubric criterion | Status | Pitch content to include |
|---|---|---|
| No longer than five minutes | MISSING | No pitch artifact exists yet. Target 4:00-4:30. |
| Online presentation supplied as accessible video | MISSING | Upload with captions and verify sharing permissions. |
| Identifies the fine-tuned model | READY FOR PITCH | State `Qwen/Qwen2.5-1.5B-Instruct`. |
| Explains fine-tuning strategy | READY FOR PITCH | Explain rank-8 LoRA via PEFT, frozen base weights, adapter-only checkpoint, and 500-example synthetic dataset. |
| Includes simple before/after demonstrating learned capability | READY FOR PITCH | Use held-out example `synthetic-0287-1476ed96ed01` and the README results table. Show the actual Streamlit side-by-side view. |
| Reflects on risks, ethics, or evaluation challenges | READY FOR PITCH | Cover hallucination, lost uncertainty, negation, over-generation, synthetic-data limits, privacy/HIPAA, bias/accessibility, and why ROUGE/readability do not establish clinical safety. |

## Application

| Rubric criterion | Status | Evidence or action |
|---|---|---|
| Live deployed web/mobile application | MISSING | Streamlit UI exists locally but no public deployment was found. |
| Runs inference on a trained model | COVERED LOCALLY | `main.py` loads the base model plus `models/medexplain_lora_adapter`; the adapter and 50 fine-tuned predictions exist locally. Must be verified again after deployment. |
| Good UX and usability | PARTIAL | Side-by-side output, examples, readability, flags, and disclaimer are implemented. Conduct a rendered browser review and user smoke test. |
| Stable | PARTIAL | Unit tests pass, but deployed cold start, memory usage, inference latency, error behavior, and concurrency have not been tested. |
| Accessible | PARTIAL | Standard Streamlit controls have labels, but keyboard navigation, screen reader behavior, contrast, zoom, and mobile layout have not been audited. |

## Git repository

| Rubric criterion | Status | Evidence or action |
|---|---|---|
| Uses branches | MISSING | The current folder is not a Git repository. Initialize Git and do remaining work on a feature branch. |
| Uses pull requests before merging to main | MISSING | Publish the feature branch and merge it through a pull request. |
| Descriptive README with run instructions | COVERED | `README.md` includes purpose, architecture, setup, exact commands, evaluation, limitations, ethics, deployment guidance, and disclaimer. |
| Dependency file | COVERED | `requirements.txt` is present with version constraints and optional Linux `bitsandbytes`. |
| Dataset creation script | COVERED | `scripts/make_dataset.py`. |
| Feature-building script | COVERED | `scripts/build_features.py`. |
| Training and prediction script | COVERED | `scripts/model.py`, orchestrated by `main.py`. |
| Notebooks only in notebooks directory | COVERED | No notebooks are used; the `notebooks/` directory exists. |
| Modular classes/functions and no loose executable code | COVERED | Pipeline logic is organized into functions with guarded CLI entry points. |
| Functions/classes include docstrings | COVERED | Public pipeline functions include docstrings. |

## Highest-priority completion order

1. Ask whether the recommendation-system lines apply to this generative-model hackathon; do not turn MedExplain into a recommender without confirmation.
2. Initialize Git, create a feature branch, publish to GitHub, and merge through a pull request.
3. Decide how deployment obtains the 8.7 MB LoRA adapter. It is currently ignored by Git; include it if permitted or download it from a documented model repository at startup.
4. Deploy the app and test trained inference, cold start, latency, errors, accessibility, and logged-out access.
5. Record a captioned pitch under five minutes using the actual held-out before/after example and honest evaluation results.
6. Test every submitted link in a private browser window before submission.

## Suggested four-minute pitch outline

- 0:00-0:30: Medical-language access problem and strict rewrite-only scope.
- 0:30-1:05: Qwen 1.5B base model, 500 synthetic examples, and why LoRA/PEFT.
- 1:05-2:10: Live side-by-side held-out before/after demonstration.
- 2:10-2:50: ROUGE and safety-heuristic results, including what improved.
- 2:50-3:35: Failures: over-generation, negation flags, synthetic-data limitations, and lack of clinical validation.
- 3:35-4:00: Responsible next steps and educational-use disclaimer.
