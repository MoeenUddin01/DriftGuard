from pathlib import Path
from typing import Dict, Optional, Union
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extractor for computing summary statistics and file metadata on datasets."""

    @staticmethod
    def extract(df: pd.DataFrame, file_path: Optional[Union[str, Path]] = None) -> Dict:
        """Extract metadata summary dictionary from a DataFrame safely.

        Args:
            df: Target DataFrame.
            file_path: Optional source file path.

        Returns:
            Dict: Summary metadata including row count, column count, columns, missing values.
        """
        try:
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Expected pandas DataFrame, got {type(df).__name__}")

            meta = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "total_null_values": int(df.isnull().sum().sum()),
            }

            if file_path:
                try:
                    path = Path(file_path)
                    meta["file_name"] = path.name
                    meta["file_size_bytes"] = path.stat().st_size if path.exists() else 0
                except Exception as file_err:
                    logger.warning(f"Failed to extract file stats for '{file_path}': {file_err}")
                    meta["file_name"] = str(file_path)
                    meta["file_size_bytes"] = 0

            return meta
        except Exception as e:
            logger.error(f"Error extracting dataset metadata: {e}")
            raise ValueError(f"Failed to extract metadata from dataset: {e}") from e
