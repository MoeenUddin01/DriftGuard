# DriftGuard

This repository contains the Loan ML System. It consists of two main components:
1. **Model Training Pipeline**: Handles data ingestion, model training, and deployment for the loan default prediction.
2. **Agentic AI**: An AI agent that triggers upon detecting drift. It investigates what changed, which feature drifted, why it happened, the impact on performance, and generates an investigation report with recommended actions for engineers.

## Architecture
- `specs/`: Specification Driven Development (SDD) files.
  - `specs/ML/`: Specifications for the ML pipeline.
  - `specs/agentic/`: Specifications for the Agentic AI.
- `src/`: Source code modules.
  - `src/data/`: Data loading and preprocessing.
  - `src/model/`: Model definition and architectures.
  - `src/pipeline/`: Training and deployment pipelines.
  - `src/agentic/`: Drift detector, investigator, and report generator agents.
