from pathlib import Path
import pytest
import pandas as pd
from src.data.data_loader import DataLoader
from src.pipeline.data_preprocessing import DataPreprocessor, save_processed_data


def test_preprocessing_pipeline():
    raw_csv = Path("dataset/raw/train_u6lujuX_CVtuZ9i.csv")
    loader = DataLoader(required_columns=["ApplicantIncome", "LoanAmount", "Loan_Status"])
    raw_df = loader.load_file(raw_csv)

    preprocessor = DataPreprocessor()
    processed_df = preprocessor.fit_transform(raw_df)

    assert isinstance(processed_df, pd.DataFrame)
    assert "Loan_Status" in processed_df.columns
    # Check target mapping (Y->1, N->0)
    assert set(processed_df["Loan_Status"].dropna().unique()).issubset({0, 1})
    # Check no missing values remain in features
    feature_cols = [c for c in processed_df.columns if c != "Loan_Status"]
    assert processed_df[feature_cols].isnull().sum().sum() == 0


def test_preprocessor_serialization(tmp_path):
    preprocessor = DataPreprocessor()
    df = pd.DataFrame({
        "ApplicantIncome": [5000, 3000, 4000],
        "LoanAmount": [100, None, 150],
        "Gender": ["Male", "Female", "Male"],
        "Loan_Status": ["Y", "N", "Y"]
    })
    preprocessor.fit(df)
    
    save_path = tmp_path / "preprocessor.joblib"
    preprocessor.save_preprocessor(save_path)
    assert save_path.exists()

    loaded_preprocessor = DataPreprocessor.load_preprocessor(save_path)
    assert loaded_preprocessor.is_fitted
    transformed = loaded_preprocessor.transform(df)
    assert len(transformed) == 3


def test_dataset_partitioning(tmp_path):
    preprocessor = DataPreprocessor()
    df = pd.DataFrame({
        "ApplicantIncome": [5000, 3000, 4000, 6000, 2500, 4500, 7000, 3200, 2900, 5100],
        "LoanAmount": [100, 120, 150, 200, 90, 110, 220, 130, 95, 140],
        "Gender": ["Male", "Female", "Male", "Male", "Female", "Male", "Female", "Male", "Male", "Female"],
        "Loan_Status": ["Y", "N", "Y", "N", "Y", "N", "Y", "N", "Y", "N"]
    })
    processed = preprocessor.fit_transform(df)
    
    out_dir = tmp_path / "processed"
    train_df, test_df = save_processed_data(processed, output_dir=out_dir, test_size=0.2, val_size=0.0, random_state=42)

    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "test.parquet").exists()
    assert len(train_df) == 8
    assert len(test_df) == 2

