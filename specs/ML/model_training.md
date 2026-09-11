| Phase | Spec | Module | Priority | Depends On | Unlocks | Status |
|-------|------|--------|----------|------------|---------|--------|
| **2** | `specs/ML/model_training.md` | `src/model/`, `src/pipeline/train_pipeline.py` | P0 — Model Building | Phase 1 | Phase 3 | `[STABLE]` |

# Feature Specification: Model Training Pipeline

## Overview
The Model Training component handles machine learning model architecture selection, hyperparameter configuration, model fitting on processed training data (`dataset/processed/train.parquet`), cross-validation tuning, and model artifact serialization for production serving and evaluation.

---

### REQ-MDL-001 — Unified Loan Classifier Architecture

**Requirement:** The `LoanClassifier` class in `src/model/classifier.py` must encapsulate binary classification models with **XGBoost** (`XGBClassifier`) as the primary default model, alongside Scikit-Learn ensemble fallbacks (`RandomForestClassifier`, `GradientBoostingClassifier`).

**Rationale:** Encapsulating classifier algorithms behind a unified interface allows seamless model selection, hyperparameter customization, validation early stopping, and standardized prediction methods.

**Acceptance Criteria:**
- Implements unified `fit(X, y, eval_set=None)`, `predict(X)`, and `predict_proba(X)` methods.
- Accepts configurable parameters (`n_estimators`, `max_depth`, `learning_rate`, `random_state`, `early_stopping_rounds`, `threshold`).
- Supports validation set evaluation (`eval_set`) and early stopping for boosting models.
- Supports custom probability thresholding for default risk scoring.
- Returns structured probability outputs for class 0 (non-default) and class 1 (default).

**Dependencies:** Scikit-Learn / XGBoost, Phase 1 (`specs/ML/data_ingestion.md`)

<details>
<summary><strong>📂 Implementation Map</strong> (Required when Status is <code>[STABLE]</code>)</summary>

**Source:** [src/model/classifier.py](file:///home/moeen/projects/DriftGuard/src/model/classifier.py)

| # | Component | Type | Description |
|---|-----------|------|-------------|
| 1 | [LoanClassifier](file:///home/moeen/projects/DriftGuard/src/model/classifier.py#L11) | `class` | Main loan risk binary classification wrapper encapsulating XGBoost & Sklearn |
| 2 | [fit](file:///home/moeen/projects/DriftGuard/src/model/classifier.py#L96) | `method` | Fits underlying classifier on X and y with optional eval_set and early stopping |
| 3 | [predict_proba](file:///home/moeen/projects/DriftGuard/src/model/classifier.py#L137) | `method` | Predicts class probabilities shape (N, 2) |
| 4 | [predict](file:///home/moeen/projects/DriftGuard/src/model/classifier.py#L164) | `method` | Predicts binary labels based on configured probability threshold |

</details>

**Tests:** `tests/test_model.py::test_classifier_initialization`

**Status:** `[STABLE]`

---

### REQ-MDL-002 — Model Training & Artifact Serialization

**Requirement:** The `ModelTrainer` class in `src/model/trainer.py` must fit the `LoanClassifier` model on processed training data (`train.parquet`) and validation data (`val.parquet`), serializing artifacts (`model.joblib` and `training_config.json`) to `models/`.

**Rationale:** Automated model fitting and serialization enable reproducible model deployments, validation evaluation, and audit logging.

**Acceptance Criteria:**
- Loads processed training data (`dataset/processed/train.parquet`) and validation dataset (`val.parquet`).
- Fits `LoanClassifier` instance on features and target `loan_status`.
- Saves serialized model artifact (`model.joblib`) and metadata configuration (`training_config.json`).
- Includes error handling for missing training datasets or invalid input features.

**Dependencies:** REQ-MDL-001

<details>
<summary><strong>📂 Implementation Map</strong> (Required when Status is <code>[STABLE]</code>)</summary>

**Source:** [src/model/trainer.py](file:///home/moeen/projects/DriftGuard/src/model/trainer.py)

| # | Component | Type | Description |
|---|-----------|------|-------------|
| 1 | [ModelTrainer](file:///home/moeen/projects/DriftGuard/src/model/trainer.py#L13) | `class` | Orchestrates dataset loading, model fitting, and artifact saving |
| 2 | [train_from_file](file:///home/moeen/projects/DriftGuard/src/model/trainer.py#L97) | `method` | Loads parquet dataset file from disk and trains model with val_data_path |
| 3 | [save_artifacts](file:///home/moeen/projects/DriftGuard/src/model/trainer.py#L113) | `method` | Serializes model.joblib and training_config.json metadata |

</details>

**Tests:** `tests/test_model.py::test_model_trainer_fit_and_save`

**Status:** `[STABLE]`

---

### REQ-MDL-003 — End-to-End Training Pipeline Orchestrator

**Requirement:** The `ModelTrainingPipeline` class in `src/pipeline/train_pipeline.py` must orchestrate data loading from `dataset/processed/train.parquet` and `val.parquet`, model training via `ModelTrainer`, and artifact saving.

**Rationale:** Providing a single top-level pipeline class allows engineers or automated CLI scripts to trigger complete model training with a single function call.

**Acceptance Criteria:**
- Executes full training workflow from processed data to saved model artifacts.
- Emits execution progress logs and records training duration.
- Returns trained classifier instance and artifact file paths.

**Dependencies:** REQ-MDL-002

<details>
<summary><strong>📂 Implementation Map</strong> (Required when Status is <code>[STABLE]</code>)</summary>

**Source:** [src/pipeline/train_pipeline.py](file:///home/moeen/projects/DriftGuard/src/pipeline/train_pipeline.py)

| # | Component | Type | Description |
|---|-----------|------|-------------|
| 1 | [ModelTrainingPipeline](file:///home/moeen/projects/DriftGuard/src/pipeline/train_pipeline.py#L12) | `class` | Top-level execution pipeline class for model training |
| 2 | [run](file:///home/moeen/projects/DriftGuard/src/pipeline/train_pipeline.py#L36) | `method` | Executes end-to-end training and artifact saving |

</details>

**Tests:** `tests/test_model.py::test_training_pipeline_end_to_end`

**Status:** `[STABLE]`


