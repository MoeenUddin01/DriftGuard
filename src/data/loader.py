from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import pandas as pd

from src.data.schema import SchemaValidator, SchemaValidationError
from src.data.metadata import MetadataExtractor

logger = logging.getLogger(__name__)


class DataLoader:
    """Loader for customer loan application datasets from CSV, Parquet, or DataFrames."""

    def __init__(self, required_columns: Optional[List[str]] = None):
        self.validator = SchemaValidator(required_columns)
        self.extractor = MetadataExtractor()

    @property
    def required_columns(self) -> List[str]:
        return self.validator.required_columns

    def load_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load dataset from a CSV or Parquet file path and validate its schema.

        Args:
            file_path: Path to the dataset file.

        Returns:
            pd.DataFrame: Loaded dataset.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If file format is unsupported or parsing fails.
            SchemaValidationError: If required columns are missing.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {path}")

        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            elif path.suffix.lower() in [".parquet", ".pq"]:
                df = pd.read_parquet(path)
            else:
                raise ValueError(
                    f"Unsupported file format '{path.suffix}'. Supported formats are .csv and .parquet"
                )
        except (ValueError, FileNotFoundError):
            raise
        except Exception as e:
            logger.error(f"Failed to parse dataset file '{path}': {e}")
            raise ValueError(f"Error reading file '{path.name}': {e}") from e

        self.validate_schema(df)
        return df

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Check if all required columns are present in the DataFrame."""
        return self.validator.validate(df)

    def get_metadata(self, df: pd.DataFrame, file_path: Optional[Union[str, Path]] = None) -> Dict:
        """Extract dataset metadata summary dictionary."""
        return self.extractor.extract(df, file_path)
