| Phase | Spec | Module | Priority | Depends On | Unlocks | Status |
|-------|------|--------|----------|------------|---------|--------|
| **5** | `specs/agentic/investigator.md` | `src/agentic/investigator.py` | P0 — AI Agent Core | Phase 4 | Phase 6 | `[STABLE]` |

# Feature Specification: AI Investigator Agent

## Overview
The AI Investigator Agent processes `DriftIncidentPayload` notifications, inspects shifted feature distributions, analyzes root causes (why the shift happened), estimates model accuracy degradation impact, and formulates diagnostic hypotheses.

---

### REQ-INV-001 — Feature Localization & Shift Analysis

**Requirement:** The `Investigator` agent in `src/agentic/investigator.py` must analyze the `DriftIncidentPayload` to pinpoint exact features that drifted and quantify the direction/magnitude of statistical shifts.

**Rationale:** Engineers need precise feature-level localization to understand customer demographic or financial pattern changes.

**Acceptance Criteria:**
- Ranks drifted features by PSI and KS statistic magnitude.
- Calculates directional mean/median shift percentages (e.g. `person_income` decreased by $18.4\%$, `loan_amnt` increased by $22.1\%$).
- Formats a feature drift summary diagnostic table.

**Dependencies:** REQ-DFT-002

<details>
<summary><strong>📂 Implementation Map</strong> (Required when Status is <code>[STABLE]</code>)</summary>

**Source:** [src/agentic/investigator.py](file:///home/moeen/projects/DriftGuard/src/agentic/investigator.py)

| # | Component | Type | Description |
|---|-----------|------|-------------|
| 1 | [Investigator](file:///home/moeen/projects/DriftGuard/src/agentic/investigator.py#L18) | `class` | Main AI investigation agent coordinator |
| 2 | [localize_shifts](file:///home/moeen/projects/DriftGuard/src/agentic/investigator.py#L93) | `method` | Pinpoints drifted features & calculates shift magnitude/direction |

</details>

**Tests:** `tests/test_investigator.py::test_feature_localization`

**Status:** `[STABLE]`

---

### REQ-INV-002 — Root Cause Reasoning & Impact Estimation

**Requirement:** The `Investigator` agent must evaluate model feature importance weights to estimate prediction accuracy loss (F1/ROC-AUC drop) and generate natural language hypotheses explaining root causes.

**Rationale:** Understanding "Why" drift occurred and its financial impact allows engineers to prioritize remediation actions.

**Acceptance Criteria:**
- Cross-references drifted features against model feature importance scores to estimate risk impact.
- Generates structured diagnostic hypotheses (e.g. macroeconomic shifts, new customer acquisition channel, data pipeline ingestion bug).
- Quantifies estimated financial risk / default misclassification risk increase.

**Dependencies:** REQ-INV-001, LLM Inference / Agent Tools

<details>
<summary><strong>📂 Implementation Map</strong> (Required when Status is <code>[STABLE]</code>)</summary>

**Source:** [src/agentic/investigator.py](file:///home/moeen/projects/DriftGuard/src/agentic/investigator.py)

| # | Component | Type | Description |
|---|-----------|------|-------------|
| 1 | [analyze_root_cause](file:///home/moeen/projects/DriftGuard/src/agentic/investigator.py#L149) | `method` | Evaluates feature importances, estimates ROC-AUC drop & formulates hypotheses |
| 2 | [investigate](file:///home/moeen/projects/DriftGuard/src/agentic/investigator.py#L217) | `method` | Orchestrates localization, root-cause analysis & recommendations |

</details>

**Tests:** `tests/test_investigator.py::test_root_cause_reasoning`, `tests/test_investigator.py::test_investigate_full_workflow`

**Status:** `[STABLE]`
