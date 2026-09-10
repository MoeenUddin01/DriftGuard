| Phase | Spec | Module | Priority | Depends On | Unlocks | Status |
|-------|------|--------|----------|------------|---------|--------|
| **2** | `specs/ML/model_training.md` | `src/model/`, `src/pipeline/model_training.py` | P0 — Model Building | Phase 1 | Phase 3 | `[PLANNED]` |

# Feature Specification: Model Training Pipeline

## Overview
The Model Training component handles model selection, hyperparameter configuration, model fitting on processed training data, and serializing model artifacts for production serving and evaluation.

---

### REQ-MDL-001 — Model Architecture & Hyperparameter Configuration

**Requirement:** The `LoanClassifier` class in `src/model/model.py` must encapsulate binary classification model architectures (e.g. Random Forest, XGBoost, or Logistic Regression) configured via external hyperparameter dictionaries.

**Rationale:** Encapsulating classifier algorithms behind a unified interface allows easy model comparison and hyperparameter tuning.

**Acceptance Criteria:**
- Implements `fit`, `predict`, and `predict_proba` unified methods.
- Accepts configurable parameters (e.g. `n_estimators`, `max_depth`, `learning_rate`, `random_state`).
- Supports probability threshold adjustment for custom default risk scoring.

**Dependencies:** Scikit-Learn / XGBoost

<details>
<summary><strong>📂 Implementation Map</strong> (Required when Status is <code>[STABLE]</code>)</summary>

**Source:** [src/model/model.py](file:///home/moeen/projects/DriftGuard/src/model/model.py)

| # | Component | Type | Description |
|---|-----------|------|-------------|
| 1 | [LoanClassifier](file:///home/moeen/projects/DriftGuard/src/model/model.py#L1) | `class` | Main loan risk binary classifier wrapper |

</details>

**Tests:** `tests/test_model.py::test_model_initialization`

**Status:** `[PLANNED]`

---

### REQ-MDL-002 — Training Pipeline & Artifact Serialization

**Requirement:** The `ModelTrainer` in `src/pipeline/model_training.py` must orchestrate model training on train datasets and serialize trained model artifacts (`model.joblib`) to `src/model/artifacts/`.

**Rationale:** Automated pipeline execution and artifact persistence enable reproducible model deployments and audit logging.

**Acceptance Criteria:**
- Loads processed training data (`dataset/processed/train.parquet`).
- Fits `LoanClassifier` instance on training partition.
- Saves serialized model object (`model.joblib`) and training metadata (`training_config.json`) to model artifact directory.
- Logs training execution duration and data shape.

**Dependencies:** REQ-DAT-003, REQ-MDL-001

<details>
<summary><strong>📂 Implementation Map</strong> (Required when Status is <code>[STABLE]</code>)</summary>

**Source:** [src/pipeline/model_training.py](file:///home/moeen/projects/DriftGuard/src/pipeline/model_training.py)

| # | Component | Type | Description |
|---|-----------|------|-------------|
| 1 | [ModelTrainer](file:///home/moeen/projects/DriftGuard/src/pipeline/model_training.py#L1) | `class` | Training orchestration and artifact saver |

</details>

**Tests:** `tests/test_model.py::test_train_pipeline`

**Status:** `[PLANNED]`
