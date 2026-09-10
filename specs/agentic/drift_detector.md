| Phase | Spec | Module | Priority | Depends On | Unlocks | Status |
|-------|------|--------|----------|------------|---------|--------|
| **4** | `specs/agentic/drift_detector.md` | `src/agentic/drift_detector.py` | P0 — Drift Trigger | Phase 3 | Phase 5 | `[PLANNED]` |

# Feature Specification: Statistical Drift Detector

## Overview
The Drift Detector component monitors incoming customer inference data, calculates statistical distribution shifts against the saved reference baseline (`dataset/baseline_stats.json`), and triggers the AI Investigator agent when significant feature or concept drift is detected.

---

### REQ-DFT-001 — Statistical Distribution Monitoring

**Requirement:** The `DriftDetector` class in `src/agentic/drift_detector.py` must compute statistical drift metrics (Kolmogorov-Smirnov test for numerical features, Chi-Square / PSI for categorical features) between incoming customer batches and baseline data.

**Rationale:** Automated statistical testing identifies early signs of data drift before severe performance degradation impacts loan default predictions.

**Acceptance Criteria:**
- Loads reference baseline distributions from `dataset/baseline_stats.json`.
- Runs Kolmogorov-Smirnov (KS) test on numerical features (e.g. `person_income`, `loan_amnt`).
- Computes Population Stability Index (PSI) for all features.
- Flags feature drift if KS p-value $< 0.05$ or $\text{PSI} > 0.2$.

**Dependencies:** REQ-EVL-002, SciPy / Evidently

<details>
<summary><strong>📂 Implementation Map</strong> (Required when Status is <code>[STABLE]</code>)</summary>

**Source:** [src/agentic/drift_detector.py](file:///home/moeen/projects/DriftGuard/src/agentic/drift_detector.py)

| # | Component | Type | Description |
|---|-----------|------|-------------|
| 1 | [DriftDetector](file:///home/moeen/projects/DriftGuard/src/agentic/drift_detector.py#L1) | `class` | Statistical drift calculator and detector |

</details>

**Tests:** `tests/test_agentic.py::test_drift_detection`

**Status:** `[PLANNED]`

---

### REQ-DFT-002 — Drift Event Payload & AI Agent Triggering

**Requirement:** When statistical drift is detected, `DriftDetector` must format a structured `DriftIncidentPayload` and dispatch it to the `Investigator` AI agent.

**Rationale:** Automated payload dispatch eliminates manual monitoring overhead and enables immediate agentic investigation.

**Acceptance Criteria:**
- Formats payload containing: `incident_id`, `timestamp`, `drifted_features`, `drift_scores` (KS/PSI values), `batch_sample_size`.
- Triggers `Investigator.investigate(payload)` asynchronously or synchronously.
- Emits drift notification logs to operational monitoring channels.

**Dependencies:** REQ-DFT-001

<details>
<summary><strong>📂 Implementation Map</strong> (Required when Status is <code>[STABLE]</code>)</summary>

**Source:** [src/agentic/drift_detector.py](file:///home/moeen/projects/DriftGuard/src/agentic/drift_detector.py)

| # | Component | Type | Description |
|---|-----------|------|-------------|
| 1 | [trigger_investigation](file:///home/moeen/projects/DriftGuard/src/agentic/drift_detector.py#L1) | `method` | Formats payload & invokes AI Investigator |

</details>

**Tests:** `tests/test_agentic.py::test_drift_trigger`

**Status:** `[PLANNED]`
