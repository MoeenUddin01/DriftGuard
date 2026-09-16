| Phase | Spec | Module | Priority | Depends On | Unlocks | Status |
|-------|------|--------|----------|------------|---------|--------|
| **6** | `specs/agentic/report_generator.md` | `src/agentic/report_generator.py` | P1 — Engineering Reports | Phase 5 | Production Integration | `[STABLE]` |

# Feature Specification: Investigation Report Generator

## Overview
The Report Generator component compiles diagnostic findings from the AI Investigator into structured, actionable Markdown Investigation Reports and exports them for engineering review and automated alerting.

---

### REQ-REP-001 — Markdown Investigation Report Compilation

**Requirement:** The `ReportGenerator` class in `src/agentic/report_generator.py` must render investigation findings into a standard Markdown report schema answering: What changed? Which feature? Why? Did performance drop? What's the impact? What should engineers do?

**Rationale:** Standardized reporting ensures engineering teams receive clear, actionable insights without needing to interpret raw statistical outputs.

**Acceptance Criteria:**
- Generates report sections:
  1. **Executive Summary** (Incident ID, Timestamp, Alert Severity).
  2. **Drifted Features Analysis** (Ranked table of shifted features & percentage shifts).
  3. **Root Cause Analysis** (Diagnosed reasons for statistical shift).
  4. **Performance & Financial Impact** (Estimated accuracy/ROC-AUC degradation).
  5. **Recommended Engineering Actions** (Actionable steps like retrain on recent window, adjust probability thresholds, audit upstream feature pipeline).
- Validates Markdown syntax compliance.

**Dependencies:** REQ-INV-002

<details>
<summary><strong>📂 Implementation Map</strong> (Required when Status is <code>[STABLE]</code>)</summary>

**Source:** [src/agentic/report_generator.py](file:///home/moeen/projects/DriftGuard/src/agentic/report_generator.py)

| # | Component | Type | Description |
|---|-----------|------|-------------|
| 1 | [ReportGenerator](file:///home/moeen/projects/DriftGuard/src/agentic/report_generator.py#L35) | `class` | Formats and compiles investigation report |
| 2 | [generate_report](file:///home/moeen/projects/DriftGuard/src/agentic/report_generator.py#L58) | `method` | Renders Markdown report document answering the 5 core diagnostic questions |

</details>

**Tests:** `tests/test_report_generator.py::test_report_generation`

**Status:** `[STABLE]`

---

### REQ-REP-002 — Report Export & Persistence

**Requirement:** The pipeline must persist rendered reports to `reports/drift_report_<timestamp>.md` and dispatch alerts to operational channels.

**Rationale:** Persisting historical drift reports creates an audit trail for compliance and tracking model stability over time.

**Acceptance Criteria:**
- Writes rendered report file to `reports/drift_report_<timestamp>.md`.
- Generates a summary notification log for console/alert integrations.

**Dependencies:** REQ-REP-001

<details>
<summary><strong>📂 Implementation Map</strong> (Required when Status is <code>[STABLE]</code>)</summary>

**Source:** [src/agentic/report_generator.py](file:///home/moeen/projects/DriftGuard/src/agentic/report_generator.py)

| # | Component | Type | Description |
|---|-----------|------|-------------|
| 1 | [save_report](file:///home/moeen/projects/DriftGuard/src/agentic/report_generator.py#L16) | `function` | Writes report markdown file to disk |
| 2 | [save_report](file:///home/moeen/projects/DriftGuard/src/agentic/report_generator.py#L198) | `method` | ReportGenerator method writing report markdown file to disk |

</details>

**Tests:** `tests/test_report_generator.py::test_report_export`

**Status:** `[STABLE]`
