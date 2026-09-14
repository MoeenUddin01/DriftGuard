import json
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from src.agentic.drift_detector import DriftDetector, DriftIncidentPayload
from src.agentic.investigator import Investigator


@pytest.fixture
def mock_baseline_stats():
    """Fixture providing synthetic reference baseline stats."""
    np.random.seed(42)
    income_samples = np.random.normal(loc=50000, scale=10000, size=500)
    loan_samples = np.random.normal(loc=15000, scale=5000, size=500)

    return {
        "num_samples": 500,
        "features": {
            "income": {
                "type": "numerical",
                "mean": float(np.mean(income_samples)),
                "std": float(np.std(income_samples)),
                "min": float(np.min(income_samples)),
                "25%": float(np.percentile(income_samples, 25)),
                "50%": float(np.percentile(income_samples, 50)),
                "75%": float(np.percentile(income_samples, 75)),
                "max": float(np.max(income_samples)),
                "deciles": [float(np.percentile(income_samples, p)) for p in range(0, 101, 10)],
                "quantiles_100": [float(np.percentile(income_samples, p)) for p in range(0, 101, 1)],
            },
            "loan_amount": {
                "type": "numerical",
                "mean": float(np.mean(loan_samples)),
                "std": float(np.std(loan_samples)),
                "min": float(np.min(loan_samples)),
                "25%": float(np.percentile(loan_samples, 25)),
                "50%": float(np.percentile(loan_samples, 50)),
                "75%": float(np.percentile(loan_samples, 75)),
                "max": float(np.max(loan_samples)),
                "deciles": [float(np.percentile(loan_samples, p)) for p in range(0, 101, 10)],
                "quantiles_100": [float(np.percentile(loan_samples, p)) for p in range(0, 101, 1)],
            },
            "home_ownership": {
                "type": "categorical",
                "frequencies": {"RENT": 0.50, "MORTGAGE": 0.40, "OWN": 0.10},
                "unique_categories": 3,
            },
        },
    }


def test_no_drift(mock_baseline_stats):
    """Test that baseline-like batch reports no drift."""
    np.random.seed(42)
    clean_batch = pd.DataFrame({
        "income": np.random.normal(loc=50000, scale=10000, size=200),
        "loan_amount": np.random.normal(loc=15000, scale=5000, size=200),
        "home_ownership": np.random.choice(["RENT", "MORTGAGE", "OWN"], size=200, p=[0.50, 0.40, 0.10]),
    })

    mock_investigator = MagicMock(spec=Investigator)
    detector = DriftDetector(baseline_source=mock_baseline_stats, investigator=mock_investigator)

    has_drift, payload, scores = detector.detect(clean_batch)

    assert has_drift is False
    assert len(payload.drifted_features) == 0
    assert payload.batch_sample_size == 200
    mock_investigator.investigate.assert_not_called()


def test_numerical_drift(mock_baseline_stats):
    """Test detecting significant numerical drift in income feature."""
    np.random.seed(42)
    drifted_batch = pd.DataFrame({
        # Mean shifted from 50,000 to 85,000
        "income": np.random.normal(loc=85000, scale=12000, size=200),
        "loan_amount": np.random.normal(loc=15000, scale=5000, size=200),
        "home_ownership": np.random.choice(["RENT", "MORTGAGE", "OWN"], size=200, p=[0.50, 0.40, 0.10]),
    })

    mock_investigator = MagicMock(spec=Investigator)
    detector = DriftDetector(baseline_source=mock_baseline_stats, investigator=mock_investigator)

    has_drift, payload, scores = detector.detect(drifted_batch)

    assert has_drift is True
    assert "income" in payload.drifted_features
    assert scores["income"]["is_drifted"] is True
    assert scores["income"]["ks_p_value"] < 0.05
    mock_investigator.investigate.assert_called_once_with(payload)


def test_categorical_drift(mock_baseline_stats):
    """Test detecting categorical distribution drift and unseen categories."""
    np.random.seed(42)
    drifted_batch = pd.DataFrame({
        "income": np.random.normal(loc=50000, scale=10000, size=200),
        "loan_amount": np.random.normal(loc=15000, scale=5000, size=200),
        # Shifting categories drastically to 90% OWN and 10% UNSEEN_CAT
        "home_ownership": np.random.choice(["OWN", "UNSEEN_CAT"], size=200, p=[0.90, 0.10]),
    })

    mock_investigator = MagicMock(spec=Investigator)
    detector = DriftDetector(baseline_source=mock_baseline_stats, investigator=mock_investigator)

    has_drift, payload, scores = detector.detect(drifted_batch)

    assert has_drift is True
    assert "home_ownership" in payload.drifted_features
    assert scores["home_ownership"]["is_drifted"] is True
    assert scores["home_ownership"]["psi"] > 0.20
    mock_investigator.investigate.assert_called_once_with(payload)


def test_multiple_drifted_features(mock_baseline_stats):
    """Test when multiple numerical and categorical features drift concurrently."""
    np.random.seed(42)
    multi_drifted_batch = pd.DataFrame({
        "income": np.random.normal(loc=90000, scale=15000, size=200),
        "loan_amount": np.random.normal(loc=40000, scale=10000, size=200),
        "home_ownership": np.random.choice(["OWN"], size=200),
    })

    mock_investigator = MagicMock(spec=Investigator)
    detector = DriftDetector(baseline_source=mock_baseline_stats, investigator=mock_investigator)

    has_drift, payload, scores = detector.detect(multi_drifted_batch)

    assert has_drift is True
    assert set(payload.drifted_features) == {"income", "loan_amount", "home_ownership"}
    assert len(payload.drifted_features) == 3
    mock_investigator.investigate.assert_called_once_with(payload)


def test_investigator_trigger_payload_structure(mock_baseline_stats):
    """Test payload attributes and JSON serialization upon Investigator dispatch."""
    np.random.seed(42)
    drifted_batch = pd.DataFrame({
        "income": np.random.normal(loc=80000, scale=10000, size=150),
        "loan_amount": np.random.normal(loc=15000, scale=5000, size=150),
        "home_ownership": ["RENT"] * 150,
    })

    mock_investigator = MagicMock(spec=Investigator)
    detector = DriftDetector(baseline_source=mock_baseline_stats, investigator=mock_investigator)

    has_drift, payload, _ = detector.detect(drifted_batch)

    assert has_drift is True
    assert isinstance(payload, DriftIncidentPayload)
    assert payload.incident_id.startswith("inc_")
    assert payload.batch_sample_size == 150
    assert "income" in payload.drifted_features

    # Check JSON serialization
    json_str = payload.to_json()
    parsed_json = json.loads(json_str)
    assert parsed_json["incident_id"] == payload.incident_id
    assert parsed_json["batch_sample_size"] == 150


def test_edge_cases_handling(mock_baseline_stats):
    """Test handling of missing values and small/empty sample sizes."""
    # Batch with missing values (NaNs)
    df_with_nans = pd.DataFrame({
        "income": [50000.0, None, 52000.0, None, 48000.0],
        "loan_amount": [15000.0, 16000.0, None, 14000.0, None],
        "home_ownership": ["RENT", None, "MORTGAGE", None, "OWN"],
    })

    detector = DriftDetector(baseline_source=mock_baseline_stats)
    has_drift, payload, scores = detector.detect(df_with_nans)

    # Missing values should be safely ignored in computation without raising errors
    assert "income" in scores
    assert "home_ownership" in scores
