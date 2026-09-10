import pytest
import pandas as pd
from pathlib import Path
from src.data.data_loader import DataLoader, SchemaValidationError


def test_load_csv_dataset():
    raw_csv = Path("dataset/raw/train_u6lujuX_CVtuZ9i.csv")
    loader = DataLoader(required_columns=["ApplicantIncome", "LoanAmount", "Credit_History"])
    df = loader.load_file(raw_csv)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "ApplicantIncome" in df.columns
    assert "LoanAmount" in df.columns


def test_schema_validation_failure(tmp_path):
    invalid_csv = tmp_path / "invalid.csv"
    df = pd.DataFrame({"WrongColumn": [1, 2, 3]})
    df.to_csv(invalid_csv, index=False)

    loader = DataLoader(required_columns=["ApplicantIncome", "LoanAmount"])
    with pytest.raises(SchemaValidationError):
        loader.load_file(invalid_csv)


def test_get_metadata():
    loader = DataLoader()
    df = pd.DataFrame({
        "ApplicantIncome": [5000, 3000],
        "LoanAmount": [100, None],
        "Credit_History": [1, 0]
    })
    meta = loader.get_metadata(df)

    assert meta["row_count"] == 2
    assert meta["column_count"] == 3
    assert meta["total_null_values"] == 1
