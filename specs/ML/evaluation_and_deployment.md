| Phase | Spec | Module | Priority | Depends On | Unlocks | Status |
|-------|------|--------|----------|------------|---------|--------|
| **3** | `specs/ML/evaluation_and_deployment.md` | `src/pipeline/model_evaluation.py`, `src/model/evaluation.py` | P0 — Evaluation & Deploy | Phase 2 | Phase 4 | `[STABLE]` |

# Feature Specification: Model Evaluation & Baseline Logging

## Overview
The Evaluation & Deployment component calculates model performance metrics against test datasets, checks baseline performance thresholds, logs baseline feature distributions, and promotes valid models to production serving status.

---

### REQ-EVL-001 — Classification Metrics Calculation

**Requirement:** The `ModelEvaluator` class in `src/model/evaluation.py` must calculate ROC-AUC, F1-Score, Precision, Recall, Accuracy, and Confusion Matrix metrics on test splits.

**Rationale:** Multi-metric evaluation ensures the model maintains a balanced risk-benefit trade-off for loan default predictions.

**Acceptance Criteria:**
- Evaluates model predictions against ground truth `loan_status` on test data split.
- Returns a comprehensive evaluation dictionary containing ROC-AUC, F1, Precision, Recall, and Confusion Matrix.
- Verifies ROC-AUC meets minimum production acceptance threshold ($\ge 0.75$).

**Dependencies:** REQ-MDL-002, Scikit-Learn

<details>
<summary><strong>📂 Implementation Map</strong> (Required when Status is <code>[STABLE]</code>)</summary>

**Source:** [src/model/evaluation.py](file:///home/moeen/projects/DriftGuard/src/model/evaluation.py)

| # | Component | Type | Description |
|---|-----------|------|-------------|
| 1 | [ModelEvaluator](file:///home/moeen/projects/DriftGuard/src/model/evaluation.py#L17) | `class` | Evaluates classification performance metrics |

</details>

**Tests:** `tests/test_evaluation.py::test_model_evaluator`

**Status:** `[STABLE]`

---

### REQ-EVL-002 — Baseline Distribution Logging & Deployment

**Requirement:** The pipeline must log baseline feature distributions (mean, std, percentiles, categorical frequencies) and serialize reference stats to `dataset/baseline_stats.json` upon deployment.

**Rationale:** Reference baseline statistics are critical for downstream statistical drift comparisons when new customer data arrives.

**Acceptance Criteria:**
- Computes baseline statistical profiles for all numerical and categorical features.
- Saves `baseline_stats.json` containing baseline metric scores and feature statistics.
- Updates production model symlink to point to the validated model artifact.

**Dependencies:** REQ-EVL-001

<details>
<summary><strong>📂 Implementation Map</strong> (Required when Status is <code>[STABLE]</code>)</summary>

**Source:** [src/pipeline/model_evaluation.py](file:///home/moeen/projects/DriftGuard/src/pipeline/model_evaluation.py)

| # | Component | Type | Description |
|---|-----------|------|-------------|
| 1 | [log_baseline_stats](file:///home/moeen/projects/DriftGuard/src/pipeline/model_evaluation.py#L14) | `function` | Computes & exports reference baseline distribution |
| 2 | [ModelEvaluationPipeline](file:///home/moeen/projects/DriftGuard/src/pipeline/model_evaluation.py#L75) | `class` | End-to-end evaluation & baseline logging orchestrator |

</details>

**Tests:** `tests/test_evaluation.py::test_log_baseline_stats`

**Status:** `[STABLE]`
