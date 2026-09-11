from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest

from src.model.classifier import LoanClassifier
from src.model.trainer import ModelTrainer
from src.pipeline.train_pipeline import ModelTrainingPipeline


@pytest.fixture
def dummy_train_df():
    """Fixture generating synthetic preprocessed loan dataset."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "ApplicantIncome": np.random.randn(n),
            "CoapplicantIncome": np.random.randn(n),
            "LoanAmount": np.random.randn(n),
            "Gender_Male": np.random.choice([0, 1], size=n),
            "Married_Yes": np.random.choice([0, 1], size=n),
            "Loan_Status": np.random.choice([0, 1], size=n),
        }
    )
    return df


def test_classifier_initialization(dummy_train_df):
    """Test LoanClassifier initialization, fitting, predictions, probability shape, and thresholding."""
    X = dummy_train_df.drop(columns=["Loan_Status"])
    y = dummy_train_df["Loan_Status"]

    # 1. XGBoost Default
    clf = LoanClassifier(model_type="xgboost", n_estimators=10, max_depth=3)
    assert not clf.is_fitted
    clf.fit(X, y)
    assert clf.is_fitted

    proba = clf.predict_proba(X)
    assert proba.shape == (100, 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))

    preds_50 = clf.predict(X)
    assert len(preds_50) == 100
    assert set(np.unique(preds_50)).issubset({0, 1})

    # Custom threshold
    clf.threshold = 0.8
    preds_80 = clf.predict(X)
    assert np.sum(preds_80) <= np.sum(preds_50)

    # 2. Random Forest and Gradient Boosting fallbacks
    for mtype in ["random_forest", "gradient_boosting"]:
        clf_other = LoanClassifier(model_type=mtype, n_estimators=5)
        clf_other.fit(X, y)
        assert clf_other.predict_proba(X).shape == (100, 2)

    # Unsupported model error
    with pytest.raises(ValueError, match="Unsupported model_type"):
        LoanClassifier(model_type="invalid_model")


def test_model_trainer_fit_and_save(dummy_train_df, tmp_path):
    """Test ModelTrainer dataset loading, training with validation set, artifact serialization, and error handling."""
    data_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    dummy_train_df.to_parquet(data_path, index=False)
    dummy_train_df.iloc[:20].to_parquet(val_path, index=False)

    trainer = ModelTrainer(target_col="Loan_Status")
    clf = trainer.train_from_file(data_path, val_data_path=val_path)
    assert clf.is_fitted

    output_dir = tmp_path / "artifacts"
    saved = trainer.save_artifacts(output_dir=output_dir)

    assert saved["model_path"].exists()
    assert saved["config_path"].exists()

    # Verify model artifact loads back
    loaded_clf = joblib.load(saved["model_path"])
    assert loaded_clf.is_fitted
    X = dummy_train_df.drop(columns=["Loan_Status"])
    assert len(loaded_clf.predict(X)) == 100

    # Verify JSON config content
    with open(saved["config_path"], "r") as f:
        config = json.load(f)
    assert config["model_type"] == "xgboost"
    assert config["target_column"] == "Loan_Status"

    # Error handling tests
    with pytest.raises(FileNotFoundError):
        trainer.train_from_file(tmp_path / "non_existent.parquet")

    with pytest.raises(ValueError, match="Target column 'Missing_Col' not found"):
        bad_trainer = ModelTrainer(target_col="Missing_Col")
        bad_trainer.train_from_dataframe(dummy_train_df)

    with pytest.raises(ValueError, match="empty or invalid"):
        trainer.train_from_dataframe(pd.DataFrame())


def test_training_pipeline_end_to_end(dummy_train_df, tmp_path):
    """Test end-to-end execution of ModelTrainingPipeline with validation partition."""
    data_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    dummy_train_df.to_parquet(data_path, index=False)
    dummy_train_df.iloc[:20].to_parquet(val_path, index=False)
    artifact_dir = tmp_path / "models"

    pipeline = ModelTrainingPipeline(model_type="xgboost", n_estimators=10)
    results = pipeline.run(
        processed_data_path=data_path, val_data_path=val_path, artifact_dir=artifact_dir
    )

    assert "classifier" in results
    assert "artifact_paths" in results
    assert "execution_time_seconds" in results

    assert results["classifier"].is_fitted
    assert results["artifact_paths"]["model_path"].exists()
    assert results["artifact_paths"]["config_path"].exists()
    assert results["execution_time_seconds"] >= 0
