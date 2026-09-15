# DriftGuard — Loan ML & Drift Monitoring Agent System

## Project Context

DriftGuard is an enterprise Loan ML System paired with an autonomous Agentic AI Drift Investigator. 
The system trains a machine learning model to predict loan default risks, continuously monitors incoming customer data for statistical drift (data drift & concept drift), and triggers an AI agent when drift occurs to investigate what changed, identify impacted features, analyze root causes, assess performance degradation, and generate actionable engineering reports.

## Tech Stack

- **Language:** Python
- **ML Frameworks:** Scikit-Learn, Pandas, NumPy
- **Agentic AI:** LangChain / AI Agent Framework
- **Monitoring & Drift:** Evidently / Custom Statistical Drift Detector

## Modular Architecture

```text
.
├── CLAUDE.md                     # AI orchestration & SDD guidelines
├── PROGRESS.md                   # Phased progress tracker & checklist
├── specs/                        # Feature specifications (Source of Truth)
│   ├── ML/
│   │   ├── data_ingestion.md     # Data acquisition, validation, & cleaning spec
│   │   ├── model_training.md     # ML algorithm selection & training pipeline spec
│   │   └── evaluation_and_deployment.md # Model evaluation & deployment spec
│   └── agentic/
│       ├── drift_detector.md     # Statistical drift detection spec
│       ├── investigator.md       # AI agent investigation workflow spec
│       └── report_generator.md   # Engineering report generation spec
├── src/
│   ├── data/                     # Dataset loading and data structures
│   │   └── data_loader.py
│   ├── model/                    # ML model definitions and evaluation logic
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluation.py
│   ├── pipeline/                 # End-to-end ML execution pipelines
│   │   ├── data_preprocessing.py
│   │   ├── model_training.py
│   │   ├── model_evaluation.py
│   │   └── train_pipeline.py
│   └── agentic/                  # Drift detector & AI agent investigation components
│       ├── drift_detector.py
│       ├── investigator.py
│       └── report_generator.py
├── dataset/                      # Raw and processed datasets
├── tests/                        # Unit and integration test suite
└── pyproject.toml                # Dependencies and project metadata
```

## Build & Run Commands

- **Environment Setup:** `python -m venv .venv`
- **Activate:** `source .venv/bin/activate`
- **Install Dependencies:** `pip install -e .`
- **Run Training Pipeline:** `python -m src.pipeline.train_pipeline`
- **Run Drift Agent:** `python -m src.agentic.drift_detector`

---

# Spec-Driven Development (SDD)

## Why this exists

This project uses Spec-Driven Development (SDD). Specifications under `specs/` are the **source of truth** for intended system behavior. Do not write code without a corresponding spec requirement.

---

## 1. Spec Status

| Status | Meaning |
| --- | --- |
| `[STABLE]` | Implemented and verified |
| `[PARTIAL]` | Implemented but incomplete or being changed |
| `[PLANNED]` | Not yet implemented |

Do not mark a requirement `[STABLE]` without verification and an **Implementation Map**.

---

## 2. Phased Execution Order

Each spec belongs to a numbered phase with clear dependencies. Phases execute in order; a phase cannot begin until its dependencies are `[STABLE]`.

| Phase | Spec | Module | Priority | Depends On | Unlocks | Status |
|-------|------|--------|----------|------------|---------|--------|
| **1** | `specs/ML/data_ingestion.md` | `src/data/`, `src/pipeline/data_preprocessing.py` | P0 — Data Pipeline | None | Phase 2 | `[STABLE]` |
| **2** | `specs/ML/model_training.md` | `src/model/`, `src/pipeline/model_training.py` | P0 — Model Building | Phase 1 | 3 | `[STABLE]` |

| **3** | `specs/ML/evaluation_and_deployment.md` | `src/pipeline/model_evaluation.py` | P0 — Evaluation & Deploy | Phase 2 | 4 | `[STABLE]` |
| **4** | `specs/agentic/drift_detector.md` | `src/agentic/drift_detector.py` | P0 — Drift Trigger | Phase 3 | 5 | `[STABLE]` |
| **5** | `specs/agentic/investigator.md` | `src/agentic/investigator.py` | P0 — AI Agent Core | Phase 4 | 6 | `[PLANNED]` |
| **6** | `specs/agentic/report_generator.md` | `src/agentic/report_generator.py` | P1 — Engineering Reports | Phase 5 | Production | `[PLANNED]` |

> **Reading a spec:** Every spec file contains a header block showing its Phase number, Priority, Module path, Dependencies, and what it Unlocks.

---

## 3. Spec Format

Every requirement in a specification uses this structure:

### REQ-XXX-NNN — Short requirement title

**Requirement:** What must be true.

**Rationale:** Why this requirement exists.

**Acceptance Criteria:**
- Concrete, checkable condition.
- Concrete, checkable condition.

**Dependencies:** Related components, requirements, or services.

<details>
<summary><strong>📂 Implementation Map</strong> (Required when Status is <code>[STABLE]</code>)</summary>

**Source:** [src/path/to/file.py](file:///path/to/file.py)

| # | Component | Type | Description |
|---|-----------|------|-------------|
| 1 | [SymbolName](file:///path/to/file.py#L12) | `class` / `method` / `function` | Short description of symbol responsibility |

</details>

**Tests:** `path/to/test.py::test_name` or `none yet`.

**Status:** `[STABLE]` / `[PARTIAL]` / `[PLANNED]`
