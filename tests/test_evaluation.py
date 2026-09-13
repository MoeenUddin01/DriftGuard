import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib
from unittest.mock import MagicMock

from src.model.evaluation import ModelEvaluator
from src.pipeline.model_evaluation import log_baseline_stats, ModelEvaluationPipeline


class DummyModel:
    """A dummy model that can be pickled by joblib."""
    def predict(self, X):
        return np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    def predict_proba(self, X):
        return np.array([
            [0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8], [0.6, 0.4],
            [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.8, 0.2], [0.1, 0.9]
        ])

@pytest.fixture
def mock_classifier():
    """Returns a picklable dummy classifier."""
    return DummyModel()


@pytest.fixture
def dummy_test_df():
    """Fixture generating synthetic test dataset."""
    df = pd.DataFrame({
        "Feature1": np.random.randn(10),
        "Feature2": np.random.randn(10),
        # Ground truth matches the mock classifier perfectly
        "Loan_Status": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    return df


@pytest.fixture
def dummy_train_df():
    """Fixture generating synthetic train dataset for baseline stats."""
    df = pd.DataFrame({
        "ApplicantIncome": [5000, 3000, 4000, None, 6000],
        "LoanAmount": [100.0, 120.0, 150.0, 200.0, 90.0],
        "Gender": ["Male", "Female", "Male", "Male", None],
        "Loan_Status": [1, 0, 1, 0, 1]
    })
    return df


def test_model_evaluator(mock_classifier, dummy_test_df):
    """Test ModelEvaluator metrics calculation and threshold validation."""
    evaluator = ModelEvaluator(target_col="Loan_Status")
    
    metrics = evaluator.evaluate(mock_classifier, dummy_test_df)
    
    assert "roc_auc" in metrics
    assert "f1_score" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics
    
    # Since our mock perfectly matches the ground truth
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0
    
    # Test threshold validation
    assert evaluator.validate_threshold(metrics, min_roc_auc=0.75) is True
    
    # Test failing threshold
    poor_metrics = {"roc_auc": 0.5}
    assert evaluator.validate_threshold(poor_metrics, min_roc_auc=0.75) is False


def test_log_baseline_stats(dummy_train_df, tmp_path):
    """Test baseline statistics logging."""
    save_path = tmp_path / "baseline_stats.json"
    
    stats = log_baseline_stats(dummy_train_df, save_path)
    
    assert save_path.exists()
    assert stats["num_samples"] == 5
    
    # Check numerical feature
    assert "ApplicantIncome" in stats["features"]
    assert stats["features"]["ApplicantIncome"]["type"] == "numerical"
    assert stats["features"]["ApplicantIncome"]["missing_count"] == 1
    
    # Check categorical feature
    assert "Gender" in stats["features"]
    assert stats["features"]["Gender"]["type"] == "categorical"
    assert "Male" in stats["features"]["Gender"]["frequencies"]
    assert stats["features"]["Gender"]["missing_count"] == 1
    
    # Target column should not be included in features
    assert "Loan_Status" not in stats["features"]


def test_evaluation_pipeline_end_to_end(mock_classifier, dummy_test_df, dummy_train_df, tmp_path):
    """Test full ModelEvaluationPipeline flow."""
    model_path = tmp_path / "model.joblib"
    joblib.dump(mock_classifier, model_path)
    
    test_path = tmp_path / "test.parquet"
    dummy_test_df.to_parquet(test_path, index=False)
    
    train_path = tmp_path / "train.parquet"
    dummy_train_df.to_parquet(train_path, index=False)
    
    baseline_path = tmp_path / "baseline_stats.json"
    
    pipeline = ModelEvaluationPipeline()
    results = pipeline.run(
        model_path=model_path,
        test_data_path=test_path,
        train_data_path=train_path,
        baseline_save_path=baseline_path,
        min_roc_auc=0.75,
    )
    
    assert results["is_valid"] is True
    assert "metrics" in results
    assert "baseline_stats_path" in results
    assert baseline_path.exists()
