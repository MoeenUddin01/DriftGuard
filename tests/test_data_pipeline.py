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
