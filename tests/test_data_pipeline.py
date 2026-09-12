from pathlib import Path
import pandas as pd
from src.pipeline.data_pipeline import DataIngestionPipeline


def test_data_ingestion_pipeline_end_to_end(tmp_path):
    raw_csv = Path("dataset/raw/train_u6lujuX_CVtuZ9i.csv")
    out_dir = tmp_path / "processed"
    preprocessor_joblib = tmp_path / "preprocessor.joblib"

    pipeline = DataIngestionPipeline()
    res = pipeline.run(
        raw_file_path=raw_csv,
        processed_dir=out_dir,
        preprocessor_save_path=preprocessor_joblib,
    )

    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "val.parquet").exists()
    assert (out_dir / "test.parquet").exists()
    assert preprocessor_joblib.exists()

    assert len(res) == 3
    train_df, val_df, test_df = res
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0


def test_no_data_leakage_in_preprocessor_fitting(tmp_path):
    """Verify that DataIngestionPipeline fits preprocessor strictly on raw training partition."""
    # Synthetic raw dataset with an extreme outlier in the test set
    raw_df = pd.DataFrame(
        {
            "ApplicantIncome": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 1000000],
            "CoapplicantIncome": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "LoanAmount": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            "Loan_Amount_Term": [360] * 10,
            "Credit_History": [1.0] * 10,
            "Gender": ["Male"] * 10,
            "Married": ["Yes"] * 10,
            "Dependents": ["0"] * 10,
            "Education": ["Graduate"] * 10,
            "Self_Employed": ["No"] * 10,
            "Property_Area": ["Urban"] * 10,
            "Loan_Status": ["Y", "N"] * 5,
        }
    )
    raw_path = tmp_path / "raw.csv"
    raw_df.to_csv(raw_path, index=False)
    out_dir = tmp_path / "processed"

    pipeline = DataIngestionPipeline()
    train_df, val_df, test_df = pipeline.run(raw_file_path=raw_path, processed_dir=out_dir)

    train_raw, val_raw, test_raw = pipeline.partitioner.split(raw_df)
    scaler = pipeline.preprocessor.transformer.named_transformers_["num"].named_steps["scaler"]
    fitted_mean = scaler.mean_[0]

    # Verify that the scaler mean matches the raw training partition mean EXACTLY,
    # and differs from the full raw dataset mean (which includes val and test partitions).
    expected_train_mean = train_raw["ApplicantIncome"].mean()
    full_dataset_mean = raw_df["ApplicantIncome"].mean()

    assert abs(fitted_mean - expected_train_mean) < 1e-5, (
        f"Preprocessor mean ({fitted_mean}) does not match train_raw mean ({expected_train_mean})."
    )
    assert abs(fitted_mean - full_dataset_mean) > 1.0, (
        f"Data leakage detected! Preprocessor mean ({fitted_mean}) matched full dataset mean ({full_dataset_mean})."
    )

