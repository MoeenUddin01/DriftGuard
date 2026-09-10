from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd


class SchemaValidationError(ValueError):
    """Raised when the dataset fails schema validation."""
    pass


class DataLoader:
    """Loader for customer loan application datasets from CSV, Parquet, or DataFrames."""

    DEFAULT_REQUIRED_COLUMNS = [
        "ApplicantIncome",
        "LoanAmount",
        "Credit_History",
    ]

    def __init__(self, required_columns: Optional[List[str]] = None):
        """Initialize DataLoader with optional schema column requirements.

        Args:
            required_columns: List of column names that must be present in the dataset.
        """
        self.required_columns = required_columns or self.DEFAULT_REQUIRED_COLUMNS

    def load_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load dataset from a CSV or Parquet file path and validate its schema.

        Args:
            file_path: Path to the dataset file.

        Returns:
            pd.DataFrame: Loaded dataset.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If file format is unsupported.
            SchemaValidationError: If required columns are missing.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {path}")

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() in [".parquet", ".pq"]:
            df = pd.read_parquet(path)
        else:
            raise ValueError(
                f"Unsupported file format '{path.suffix}'. Supported formats are .csv and .parquet"
            )

        self.validate_schema(df)
        return df

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Check if all required columns are present in the DataFrame.

        Args:
            df: DataFrame to validate.

        Returns:
            bool: True if schema is valid.

        Raises:
            SchemaValidationError: If mandatory columns are missing.
        """
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise SchemaValidationError(
                f"Dataset schema validation failed. Missing required columns: {missing}"
            )
        return True

    def get_metadata(self, df: pd.DataFrame, file_path: Optional[Union[str, Path]] = None) -> Dict:
        """Extract dataset metadata summary dictionary.

        Args:
            df: Target DataFrame.
            file_path: Optional source file path.

        Returns:
            Dict: Summary metadata including row count, column count, column names, missing values.
        """
        meta = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "total_null_values": int(df.isnull().sum().sum()),
        }
        if file_path:
            path = Path(file_path)
            meta["file_name"] = path.name
            meta["file_size_bytes"] = path.stat().st_size if path.exists() else 0
        return meta
