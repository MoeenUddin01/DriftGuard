from pathlib import Path
import pytest

from src.agentic.report_generator import ReportGenerator, save_report


@pytest.fixture
def dummy_investigation_results():
    """Fixture providing synthetic investigation findings dictionary."""
    return {
        "status": "investigation_completed",
        "incident_id": "inc_20260916_report_test",
        "timestamp": "2026-09-16T09:00:00Z",
        "batch_sample_size": 250,
        "num_drifted_features": 2,
        "ranked_localized_shifts": [
            {
                "feature": "home_ownership",
                "type": "categorical",
                "psi": 0.35,
                "ks_stat": 0.0,
                "ks_p_value": 1.0,
                "direction": "shifted",
                "pct_change": None,
                "summary": "Categorical feature 'home_ownership' distribution shifted.",
            },
            {
                "feature": "person_income",
                "type": "numerical",
                "psi": 0.28,
                "ks_stat": 0.32,
                "ks_p_value": 0.002,
                "direction": "increased",
                "pct_change": 22.5,
                "summary": "Numerical feature 'person_income' mean increased by 22.5%",
            },
        ],
        "root_cause_analysis": {
            "weighted_drift_impact": 0.185,
            "estimated_roc_auc_drop": 0.0278,
            "feature_impacts": [
                {
                    "feature": "person_income",
                    "psi": 0.28,
                    "importance_weight": 0.45,
                    "impact_score": 0.126,
                },
                {
                    "feature": "home_ownership",
                    "psi": 0.35,
                    "importance_weight": 0.15,
                    "impact_score": 0.0525,
                },
            ],
            "hypotheses": [
                {
                    "hypothesis_id": "HYP-001",
                    "category": "Macroeconomic Shift",
                    "confidence": "HIGH",
                    "description": "Applicant income distributions shifted upwards due to economic changes.",
                }
            ],
            "recommended_actions": [
                "Audit upstream feature ingestion pipeline.",
                "Trigger retraining pipeline on recent customer window.",
            ],
        },
    }


def test_report_generation(dummy_investigation_results):
    """Test rendering investigation findings into structured Markdown report."""
    generator = ReportGenerator()
    report_md = generator.generate_report(dummy_investigation_results)

    # Check document title & incident ID
    assert "DriftGuard Investigation Report" in report_md
    assert "inc_20260916_report_test" in report_md

    # Check presence of all 5 required section headers
    assert "## 1. Executive Summary" in report_md
    assert "## 2. Drifted Features Analysis" in report_md
    assert "## 3. Root Cause Analysis" in report_md
    assert "## 4. Performance & Financial Impact" in report_md
    assert "## 5. Recommended Engineering Actions" in report_md

    # Check drifted feature details in Markdown tables
    assert "`person_income`" in report_md
    assert "`home_ownership`" in report_md
    assert "Macroeconomic Shift" in report_md
    assert "Audit upstream feature ingestion pipeline." in report_md


def test_report_export(dummy_investigation_results, tmp_path):
    """Test saving rendered report to disk."""
    generator = ReportGenerator(output_dir=tmp_path)
    report_md = generator.generate_report(dummy_investigation_results)

    saved_path = generator.save_report(report_md, timestamp_str="20260916_100000")

    assert saved_path.exists()
    assert saved_path.is_file()
    assert saved_path.name == "drift_report_20260916_100000.md"

    content = saved_path.read_text(encoding="utf-8")
    assert "DriftGuard Investigation Report" in content

    # Test module-level convenience function
    custom_target = tmp_path / "custom_report.md"
    result_path = save_report(report_md, output_path=custom_target)
    assert result_path.exists()
    assert result_path.name == "custom_report.md"


def test_no_drift_report():
    """Test generating a report when no features drifted."""
    no_drift_results = {
        "status": "investigation_completed",
        "incident_id": "inc_clean_001",
        "timestamp": "2026-09-16T09:00:00Z",
        "batch_sample_size": 500,
        "num_drifted_features": 0,
        "ranked_localized_shifts": [],
        "root_cause_analysis": {
            "weighted_drift_impact": 0.0,
            "estimated_roc_auc_drop": 0.0,
            "feature_impacts": [],
            "hypotheses": [],
            "recommended_actions": ["Continue standard real-time statistical monitoring."],
        },
    }

    generator = ReportGenerator()
    report_md = generator.generate_report(no_drift_results)

    assert "## 1. Executive Summary" in report_md
    assert "No significant feature drift detected" in report_md
    assert "LOW" in report_md
