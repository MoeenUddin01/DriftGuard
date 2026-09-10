| Phase | Spec | Module | Priority | Depends On | Unlocks | Status |
|-------|------|--------|----------|------------|---------|--------|
| **1** | `specs/ML/data_ingestion.md` | `src/data/`, `src/pipeline/` | P0 — Data Pipeline | None | Phase 2 | `[STABLE]` |

# Feature Specification: Data Ingestion & Preprocessing

## Overview
The Data Ingestion & Preprocessing component is responsible for loading customer loan application datasets from raw data sources (`dataset/raw/`), validating mandatory attributes (income, credit score, loan amount, etc.), cleaning missing values, encoding categorical features, scaling numerical variables, and saving processed data to `dataset/processed/`.

---

### REQ-DAT-001 — Multi-Format Loan Dataset Ingestion

**Requirement:** The `DataLoader` class must ingest raw loan datasets from `.csv`, `.parquet`, or database configurations into structured Pandas DataFrames.

**Rationale:** Standardized data loading ensures consistent inputs across training pipelines and real-time drift comparison datasets.

**Acceptance Criteria:**
- Ingests `.csv` and `.parquet` data files from `dataset/raw/`.
- Validates presence of mandatory loan columns (e.g. `person_age`, `person_income`, `loan_amnt`, `loan_intent`, `cb_person_cred_hist_length`, `loan_status`).
- Raises structured data validation errors if required schema columns are missing.
- Attaches dataset metadata (row count, feature names, file path) to the loaded object.

**Dependencies:** Pandas, PyArrow

<details>
<summary><strong>📂 Implementation Map</strong> (Required when Status is <code>[STABLE]</code>)</summary>

**Source:** [src/data/loader.py](file:///home/moeen/projects/DriftGuard/src/data/loader.py), [src/data/schema.py](file:///home/moeen/projects/DriftGuard/src/data/schema.py), [src/data/metadata.py](file:///home/moeen/projects/DriftGuard/src/data/metadata.py)

| # | Component | Type | Description |
|---|-----------|------|-------------|
| 1 | [DataLoader](file:///home/moeen/projects/DriftGuard/src/data/loader.py#L9) | `class` | Loan data ingestion and validation loader |
| 2 | [load_file](file:///home/moeen/projects/DriftGuard/src/data/loader.py#L20) | `method` | Loads CSV and Parquet files with schema validation |
| 3 | [SchemaValidator](file:///home/moeen/projects/DriftGuard/src/data/schema.py#L10) | `class` | Validates presence of required schema columns |
| 4 | [MetadataExtractor](file:///home/moeen/projects/DriftGuard/src/data/metadata.py#L6) | `class` | Generates summary statistics and file metadata |

</details>

**Tests:** [tests/test_data_loader.py::test_load_csv_dataset](file:///home/moeen/projects/DriftGuard/tests/test_data_loader.py#L7), [tests/test_data_loader.py::test_schema_validation_failure](file:///home/moeen/projects/DriftGuard/tests/test_data_loader.py#L17)

**Status:** `[STABLE]`

---

### REQ-DAT-002 — Feature Preprocessing & Cleaning Pipeline

**Requirement:** The `DataPreprocessor` class in `src/pipeline/preprocessor.py` must impute missing values, encode categorical variables, and normalize numerical features.

**Rationale:** Machine learning models require clean, numerical, and properly scaled inputs to prevent training bias and runtime exception errors.

**Acceptance Criteria:**
- Imputes missing numerical values (median strategy) and categorical values (most frequent/unknown).
- Encodes categorical attributes (`Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`) using One-Hot Encoding.
- Standardizes numerical variables using `StandardScaler`.
- Exports fitted preprocessor/scaler transformer objects to `dataset/preprocessor.joblib`.

**Dependencies:** REQ-DAT-001, Scikit-Learn

<details>
<summary><strong>📂 Implementation Map</strong> (Required when Status is <code>[STABLE]</code>)</summary>

**Source:** [src/pipeline/preprocessor.py](file:///home/moeen/projects/DriftGuard/src/pipeline/preprocessor.py)

| # | Component | Type | Description |
|---|-----------|------|-------------|
| 1 | [DataPreprocessor](file:///home/moeen/projects/DriftGuard/src/pipeline/preprocessor.py#L11) | `class` | Imputation, encoding, and scaling pipeline |
| 2 | [fit](file:///home/moeen/projects/DriftGuard/src/pipeline/preprocessor.py#L64) | `method` | Fits column transformer pipelines on features |
| 3 | [transform](file:///home/moeen/projects/DriftGuard/src/pipeline/preprocessor.py#L72) | `method` | Applies imputers, encoders, and scalers |
| 4 | [save_preprocessor](file:///home/moeen/projects/DriftGuard/src/pipeline/preprocessor.py#L97) | `method` | Serializes preprocessor via joblib |

</details>

**Tests:** [tests/test_preprocessing.py::test_preprocessing_pipeline](file:///home/moeen/projects/DriftGuard/tests/test_preprocessing.py#L7), [tests/test_preprocessing.py::test_preprocessor_serialization](file:///home/moeen/projects/DriftGuard/tests/test_preprocessing.py#L22)

**Status:** `[STABLE]`

---

### REQ-DAT-003 — Dataset Partitioning & Pipeline Orchestration

**Requirement:** Split preprocessed loan data into train/test subsets (default: 80/20) and save split datasets to `dataset/processed/` via `DataIngestionPipeline`.

**Rationale:** Separating train and test data prevents data leakage, while saving processed datasets avoids redundant preprocessing on repeated model training runs.

**Acceptance Criteria:**
- Performs stratified train/test split on `loan_status` label.
- Saves `train.parquet` and `test.parquet` to `dataset/processed/`.
- Orchestrates full workflow via `DataIngestionPipeline.run()`.

**Dependencies:** REQ-DAT-002

<details>
<summary><strong>📂 Implementation Map</strong> (Required when Status is <code>[STABLE]</code>)</summary>

**Source:** [src/pipeline/partition.py](file:///home/moeen/projects/DriftGuard/src/pipeline/partition.py), [src/pipeline/data_pipeline.py](file:///home/moeen/projects/DriftGuard/src/pipeline/data_pipeline.py)

| # | Component | Type | Description |
|---|-----------|------|-------------|
| 1 | [DatasetPartitioner](file:///home/moeen/projects/DriftGuard/src/pipeline/partition.py#L7) | `class` | Stratified train/test dataset partitioner |
| 2 | [save_processed_data](file:///home/moeen/projects/DriftGuard/src/pipeline/partition.py#L42) | `function` | Helper function for partition saving |
| 3 | [DataIngestionPipeline](file:///home/moeen/projects/DriftGuard/src/pipeline/data_pipeline.py#L9) | `class` | End-to-end data ingestion orchestrator |

</details>

**Tests:** [tests/test_preprocessing.py::test_dataset_partitioning](file:///home/moeen/projects/DriftGuard/tests/test_preprocessing.py#L40), [tests/test_data_pipeline.py::test_data_ingestion_pipeline_end_to_end](file:///home/moeen/projects/DriftGuard/tests/test_data_pipeline.py#L6)

**Status:** `[STABLE]`
