from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from src.agentic.drift_detector import DriftIncidentPayload
from src.agentic.investigator import Investigator


@pytest.fixture
def mock_baseline_stats():
    """Fixture providing baseline statistics."""
    return {
        "features": {
            "income": {
                "type": "numerical",
                "mean": 50000.0,
                "std": 10000.0,
                "min": 20000.0,
                "50%": 49000.0,
                "max": 90000.0,
            },
            "loan_amount": {
                "type": "numerical",
                "mean": 15000.0,
                "std": 5000.0,
                "min": 5000.0,
                "50%": 14000.0,
                "max": 35000.0,
            },
            "home_ownership": {
                "type": "categorical",
                "frequencies": {"RENT": 0.50, "MORTGAGE": 0.40, "OWN": 0.10},
            },
        }
    }


@pytest.fixture
def sample_payload():
    """Fixture providing a synthetic DriftIncidentPayload."""
    return DriftIncidentPayload(
        incident_id="inc_20260916_test01",
        timestamp="2026-09-16T08:00:00Z",
        drifted_features=["income", "home_ownership"],
        drift_scores={
            "income": {
                "type": "numerical",
                "ks_stat": 0.35,
                "ks_p_value": 0.001,
                "psi": 0.28,
                "is_drifted": True,
            },
            "home_ownership": {
                "type": "categorical",
                "psi": 0.45,
                "chi2_stat": 25.0,
                "chi2_p_value": 0.0001,
                "is_drifted": True,
            },
        },
        batch_sample_size=200,
    )


def test_feature_localization(mock_baseline_stats, sample_payload):
    """Test feature localization, shift direction detection, and severity ranking."""
    np.random.seed(42)
    batch_df = pd.DataFrame({
        # Income increased from mean 50,000 to 75,000
        "income": np.random.normal(loc=75000, scale=10000, size=200),
        "loan_amount": np.random.normal(loc=15000, scale=5000, size=200),
        "home_ownership": ["OWN"] * 200,
    })

    investigator = Investigator(baseline_source=mock_baseline_stats)
    localized = investigator.localize_shifts(sample_payload, batch_df=batch_df)

    assert len(localized) == 2
    # home_ownership has higher PSI (0.45) so it should be ranked first
    assert localized[0]["feature"] == "home_ownership"
    assert localized[1]["feature"] == "income"

    # Check income shift details
    income_shift = next(s for s in localized if s["feature"] == "income")
    assert income_shift["direction"] == "increased"
    assert income_shift["pct_change"] > 0.0


def test_root_cause_reasoning(mock_baseline_stats, sample_payload):
    """Test root cause reasoning and accuracy degradation estimation."""
    mock_model = MagicMock()
    mock_model.feature_importances_ = np.array([0.6, 0.3, 0.1])
    mock_model.feature_names_in_ = ["income", "loan_amount", "home_ownership"]

    investigator = Investigator(baseline_source=mock_baseline_stats, model_source=mock_model)
    localized = investigator.localize_shifts(sample_payload)
    analysis = investigator.analyze_root_cause(sample_payload, localized)

    assert "weighted_drift_impact" in analysis
    assert "estimated_roc_auc_drop" in analysis
    assert analysis["estimated_roc_auc_drop"] > 0.0
    assert len(analysis["hypotheses"]) >= 2
    assert len(analysis["recommended_actions"]) >= 2


def test_investigate_full_workflow(mock_baseline_stats, sample_payload):
    """Test end-to-end investigation workflow."""
    batch_df = pd.DataFrame({
        "income": [80000.0] * 200,
        "loan_amount": [15000.0] * 200,
        "home_ownership": ["OWN"] * 200,
    })

    investigator = Investigator(baseline_source=mock_baseline_stats)
    result = investigator.investigate(sample_payload, batch_df=batch_df)

    assert result["status"] == "investigation_completed"
    assert result["incident_id"] == sample_payload.incident_id
    assert result["num_drifted_features"] == 2
    assert len(result["ranked_localized_shifts"]) == 2
    assert "root_cause_analysis" in result


def test_edge_cases_missing_artifacts():
    """Test Investigator behavior when baseline or model file is missing."""
    # Pass nonexistent paths
    investigator = Investigator(
        baseline_source="nonexistent_baseline.json",
        model_source="nonexistent_model.joblib",
    )

    dummy_payload = DriftIncidentPayload(
        incident_id="inc_edge_001",
        timestamp="2026-09-16T08:00:00Z",
        drifted_features=["income"],
        drift_scores={"income": {"type": "numerical", "psi": 0.25, "ks_stat": 0.3, "ks_p_value": 0.01}},
        batch_sample_size=50,
    )

    result = investigator.investigate(dummy_payload)
    assert result["status"] == "investigation_completed"
    assert result["incident_id"] == "inc_edge_001"
