import pytest
from pathlib import Path
from src.utils.config import load_config, get_config_value
from src.model.classifier import LoanClassifier
from src.pipeline.train_pipeline import ModelTrainingPipeline


def test_load_config_default():
    """Test loading configuration from config.yaml in repo root."""
    config = load_config()
    assert "project" in config
    assert "data" in config
    assert "model" in config
    assert config["model"]["epochs"] == 100


def test_get_config_value():
    """Test nested dot-notation key lookup."""
    val = get_config_value("model.epochs")
    assert val == 100

    target = get_config_value("model.target_col")
    assert target == "Loan_Status"

    # Fallback to default for non-existent key
    non_existent = get_config_value("model.non_existent_key", default=42)
    assert non_existent == 42


def test_loan_classifier_epochs_alias():
    """Test that setting epochs maps to n_estimators in LoanClassifier."""
    clf = LoanClassifier(epochs=50)
    assert clf.epochs == 50
    assert clf.n_estimators == 50

    config = clf.get_config()
    assert config["epochs"] == 50
    assert config["n_estimators"] == 50


def test_model_training_pipeline_loads_config_epochs():
    """Test that ModelTrainingPipeline loads epochs default from config.yaml."""
    pipeline = ModelTrainingPipeline()
    assert pipeline.trainer.classifier.n_estimators == 100
