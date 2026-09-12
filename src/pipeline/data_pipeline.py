from pathlib import Path
from typing import Optional, Tuple, Union
import logging
import pandas as pd

from src.data.loader import DataLoader
from src.pipeline.preprocessor import DataPreprocessor
from src.pipeline.partition import DatasetPartitioner

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """Orchestrator pipeline for end-to-end data loading, preprocessing, and partitioning."""

    def __init__(
        self,
        loader: Optional[DataLoader] = None,
        preprocessor: Optional[DataPreprocessor] = None,
        partitioner: Optional[DatasetPartitioner] = None,
    ):
        self.loader = loader or DataLoader()
        self.preprocessor = preprocessor or DataPreprocessor()
        self.partitioner = partitioner or DatasetPartitioner()

    def run(
        self,
        raw_file_path: Union[str, Path],
        processed_dir: Union[str, Path],
        preprocessor_save_path: Optional[Union[str, Path]] = None,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Execute the full data ingestion and preprocessing pipeline safely.

        Args:
            raw_file_path: Path to input raw dataset CSV/Parquet file.
            processed_dir: Path to directory for saving train, val, & test parquet partitions.
            preprocessor_save_path: Optional path for saving preprocessor joblib artifact.

        Returns:
            Tuple: (train_df, val_df, test_df) or (train_df, test_df) based on partitioner configuration.

        Raises:
            Exception: Re-raises any error occurring during pipeline execution with logged context.
        """
        logger.info(f"Starting DataIngestionPipeline execution for '{raw_file_path}'...")
        try:
            raw_df = self.loader.load_file(raw_file_path)
            raw_partitions = self.partitioner.split(raw_df)

            out_dir = Path(processed_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            if len(raw_partitions) == 3:
                train_raw, val_raw, test_raw = raw_partitions
                train_df = self.preprocessor.fit_transform(train_raw)
                val_df = self.preprocessor.transform(val_raw)
                test_df = self.preprocessor.transform(test_raw)

                if preprocessor_save_path:
                    self.preprocessor.save_preprocessor(preprocessor_save_path)

                train_df.to_parquet(out_dir / "train.parquet", index=False)
                val_df.to_parquet(out_dir / "val.parquet", index=False)
                test_df.to_parquet(out_dir / "test.parquet", index=False)

                logger.info("DataIngestionPipeline executed successfully with 3-way split (train/val/test).")
                return train_df, val_df, test_df
            else:
                train_raw, test_raw = raw_partitions
                train_df = self.preprocessor.fit_transform(train_raw)
                test_df = self.preprocessor.transform(test_raw)

                if preprocessor_save_path:
                    self.preprocessor.save_preprocessor(preprocessor_save_path)

                train_df.to_parquet(out_dir / "train.parquet", index=False)
                test_df.to_parquet(out_dir / "test.parquet", index=False)

                logger.info("DataIngestionPipeline executed successfully with 2-way split (train/test).")
                return train_df, test_df
        except Exception as e:
            logger.error(f"DataIngestionPipeline execution failed: {e}")
            raise
