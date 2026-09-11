from pathlib import Path
from typing import Optional, Tuple, Union
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DatasetPartitioner:
    """Partitioner for splitting processed datasets into stratified train/val/test partitions."""

    def __init__(
        self,
        target_col: str = "Loan_Status",
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
    ):
        """Initialize DatasetPartitioner.

        Args:
            target_col: Target column name for stratification.
            test_size: Proportion of dataset for test set (0.0 to 1.0).
            val_size: Proportion of dataset for validation set (0.0 to 1.0).
            random_state: Random seed for reproducibility.
        """
        if not (0.0 <= test_size < 1.0):
            raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")
        if not (0.0 <= val_size < 1.0):
            raise ValueError(f"val_size must be between 0.0 and 1.0, got {val_size}")
        if test_size + val_size >= 1.0:
            raise ValueError(f"Sum of test_size ({test_size}) and val_size ({val_size}) must be < 1.0")

        self.target_col = target_col
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split_and_save(
        self, df: pd.DataFrame, output_dir: Union[str, Path]
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Perform train/val/test split and export Parquet files.

        Args:
            df: Input preprocessed DataFrame.
            output_dir: Output path directory.

        Returns:
            Tuple: (train_df, val_df, test_df) if val_size > 0 else (train_df, test_df).

        Raises:
            ValueError: If input dataset is invalid or splitting fails.
            IOError: If saving Parquet files fails.
        """
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            raise ValueError("Input DataFrame for partitioning is empty or invalid.")

        try:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            has_target = self.target_col in df.columns

            # Step 1: Split off test set
            stratify_test = df[self.target_col] if has_target else None
            train_val_df, test_df = train_test_split(
                df,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify_test,
            )

            # Step 2: Split train_val into train and validation if val_size > 0
            if self.val_size > 0:
                relative_val_size = self.val_size / (1.0 - self.test_size)
                stratify_val = train_val_df[self.target_col] if has_target else None
                train_df, val_df = train_test_split(
                    train_val_df,
                    test_size=relative_val_size,
                    random_state=self.random_state,
                    stratify=stratify_val,
                )
            else:
                train_df = train_val_df
                val_df = None

        except Exception as e:
            logger.error(f"Failed during dataset partitioning: {e}")
            raise ValueError(f"Dataset partitioning failed: {e}") from e

        try:
            train_df.to_parquet(out_path / "train.parquet", index=False)
            test_df.to_parquet(out_path / "test.parquet", index=False)
            
            if val_df is not None:
                val_df.to_parquet(out_path / "val.parquet", index=False)
                logger.info(
                    f"Successfully partitioned data into train ({len(train_df)}), "
                    f"val ({len(val_df)}), and test ({len(test_df)}) partitions."
                )
                return train_df, val_df, test_df
            else:
                logger.info(
                    f"Successfully partitioned data into train ({len(train_df)}) "
                    f"and test ({len(test_df)}) partitions."
                )
                return train_df, test_df

        except Exception as e:
            logger.error(f"Failed to save parquet partitions to '{output_dir}': {e}")
            raise IOError(f"Error saving parquet partitions to '{output_dir}': {e}") from e


def save_processed_data(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    target_col: str = "Loan_Status",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Helper function for splitting and saving processed dataset into train/val/test."""
    partitioner = DatasetPartitioner(
        target_col=target_col,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )
    return partitioner.split_and_save(df, output_dir)
