from pathlib import Path
from typing import Tuple, Union
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DatasetPartitioner:
    """Partitioner for splitting processed datasets into stratified train/test partitions."""

    def __init__(self, target_col: str = "Loan_Status", test_size: float = 0.2, random_state: int = 42):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

    def split_and_save(
        self, df: pd.DataFrame, output_dir: Union[str, Path]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform train/test split and export to Parquet files.

        Args:
            df: Input preprocessed DataFrame.
            output_dir: Output path directory.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)

        Raises:
            ValueError: If input dataset is invalid or splitting fails.
            IOError: If saving Parquet files fails.
        """
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            raise ValueError("Input DataFrame for partitioning is empty or invalid.")

        try:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            if self.target_col in df.columns:
                train_df, test_df = train_test_split(
                    df,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    stratify=df[self.target_col],
                )
            else:
                train_df, test_df = train_test_split(
                    df, test_size=self.test_size, random_state=self.random_state
                )
        except Exception as e:
            logger.error(f"Failed during train/test partitioning: {e}")
            raise ValueError(f"Dataset partitioning failed: {e}") from e

        try:
            train_df.to_parquet(out_path / "train.parquet", index=False)
            test_df.to_parquet(out_path / "test.parquet", index=False)
            return train_df, test_df
        except Exception as e:
            logger.error(f"Failed to save parquet partitions to '{output_dir}': {e}")
            raise IOError(f"Error saving parquet partitions to '{output_dir}': {e}") from e


def save_processed_data(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    target_col: str = "Loan_Status",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Helper function for splitting and saving processed dataset."""
    partitioner = DatasetPartitioner(
        target_col=target_col, test_size=test_size, random_state=random_state
    )
    return partitioner.split_and_save(df, output_dir)
