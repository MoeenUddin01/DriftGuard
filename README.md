# 🏦 DriftGuard

> **Autonomous Loan ML & Drift Monitoring Agent System**

DriftGuard is an enterprise machine learning system for loan default risk classification, paired with an autonomous **AI Drift Investigator Agent**. It continuously monitors incoming customer data for statistical drift (data & concept drift), automatically triggers an AI investigation agent upon drift detection, and compiles actionable root-cause engineering reports.

---

## 📐 System Workflow

```text
                 🏦 LOAN ML SYSTEM
                       │
              ┌────────▼────────┐
              │  Training Data  │
              └────────┬────────┘
                       │
                       ▼
                  Train Model
                       │
                       ▼
                 🤖 ML MODEL
                       │
              ───── DEPLOY ─────
                       │
                       ▼
              New Customer Data
                       │
                       ▼
              ┌────────────────┐
              │ Drift Detector │
              └───────┬────────┘
                      │
             ┌────────┴────────┐
             │                 │
          No Drift          Drift 🚨
             │                 │
             ↓                 ↓
          Continue         AI AGENT 🤖
                           │
                           ├── What changed?
                           ├── Which feature?
                           ├── Why?
                           ├── Did performance drop?
                           ├── What's the impact?
                           └── What should engineers do?
                                      │
                                      ↓
                              Investigation Report
```

---

## ✨ Key Features

- **Machine Learning Pipeline**: End-to-end data ingestion, validation, imputing, scaling, encoding, model training, and evaluation for loan default classification.
- **Statistical Drift Detector**: Real-time distribution monitoring using Kolmogorov-Smirnov (KS) tests and Population Stability Index (PSI) against reference baseline distributions.
- **Autonomous AI Investigator**: Root-cause reasoning agent that diagnoses shifted features, reasons about underlying causes, and projects accuracy degradation.
- **Actionable Reporting**: Auto-compiles markdown investigation reports with concrete recommendations for ML engineering teams.
- **Spec-Driven Development (SDD)**: Source-of-truth specification framework under `specs/` with strict requirement mapping (`REQ-XXX-NNN`).

---

## 📁 Repository Architecture

```text
DriftGuard/
├── CLAUDE.md                     # Agent orchestration & SDD guidelines
├── README.md                     # System documentation & setup guide
├── specs/                        # Feature specifications (Source of Truth)
│   ├── ML/
│   │   ├── data_ingestion.md     # Phase 1: Data ingestion & preprocessing spec [STABLE]
│   │   ├── model_training.md     # Phase 2: Model training & serialization spec
│   │   └── evaluation_and_deployment.md # Phase 3: Model evaluation & baseline logging spec
│   └── agentic/
│       ├── drift_detector.md     # Phase 4: Drift detection & trigger payload spec
│       ├── investigator.md       # Phase 5: AI agent root-cause analysis spec
│       └── report_generator.md   # Phase 6: Investigation report compilation spec
├── src/
│   ├── data/                     # Ingestion & schema validation
│   │   └── data_loader.py
│   ├── model/                    # ML model architectures & metrics
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluation.py
│   ├── pipeline/                 # Execution pipelines
│   │   ├── data_preprocessing.py
│   │   ├── model_training.py
│   │   ├── model_evaluation.py
│   │   └── train_pipeline.py
│   └── agentic/                  # Drift monitoring & agentic investigator
│       ├── drift_detector.py
│       ├── investigator.py
│       └── report_generator.py
├── dataset/
│   ├── raw/                      # Raw incoming customer & training datasets
│   └── processed/                # Preprocessed train/test Parquet partitions
├── tests/                        # Unit & integration test suite
└── pyproject.toml                # Dependencies & package metadata
```

---

## 🚦 Phased Development Status

This project follows **Spec-Driven Development (SDD)**. Modules are built sequentially in phases:

| Phase | Specification | Target Module | Priority | Status |
|-------|---------------|---------------|----------|--------|
| **Phase 1** | [`specs/ML/data_ingestion.md`](file:///home/moeen/projects/DriftGuard/specs/ML/data_ingestion.md) | `src/data/`, `src/pipeline/data_preprocessing.py` | P0 — Data Pipeline | `[STABLE]` |
| **Phase 2** | [`specs/ML/model_training.md`](file:///home/moeen/projects/DriftGuard/specs/ML/model_training.md) | `src/model/`, `src/pipeline/model_training.py` | P0 — Model Building | `[STABLE]` |
| **Phase 3** | [`specs/ML/evaluation_and_deployment.md`](file:///home/moeen/projects/DriftGuard/specs/ML/evaluation_and_deployment.md) | `src/pipeline/model_evaluation.py` | P0 — Evaluation & Deploy | `[STABLE]` |
| **Phase 4** | [`specs/agentic/drift_detector.md`](file:///home/moeen/projects/DriftGuard/specs/agentic/drift_detector.md) | `src/agentic/drift_detector.py` | P0 — Drift Trigger | `[PLANNED]` |
| **Phase 5** | [`specs/agentic/investigator.md`](file:///home/moeen/projects/DriftGuard/specs/agentic/investigator.md) | `src/agentic/investigator.py` | P0 — AI Agent Core | `[PLANNED]` |
| **Phase 6** | [`specs/agentic/report_generator.md`](file:///home/moeen/projects/DriftGuard/specs/agentic/report_generator.md) | `src/agentic/report_generator.py` | P1 — Engineering Reports | `[PLANNED]` |

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/MoeenUddin01/DriftGuard.git
cd DriftGuard

# Create virtual environment & install dependencies using uv
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

### 2. Running Unit Tests

```bash
# Run pytest test suite
PYTHONPATH=. .venv/bin/pytest tests/
```

### 3. Usage Example (Data Preprocessing & Ingestion)

```python
from src.pipeline.data_pipeline import DataIngestionPipeline

# Ingest, partition raw data, and fit preprocessor strictly on train partition (leakage-free)
pipeline = DataIngestionPipeline()
train_df, val_df, test_df = pipeline.run(
    raw_file_path="dataset/raw/train_u6lujuX_CVtuZ9i.csv",
    processed_dir="dataset/processed/",
    preprocessor_save_path="models/preprocessor.joblib",
)
```

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
