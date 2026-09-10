from typing import List, Optional
import pandas as pd


class SchemaValidationError(ValueError):
    """Raised when dataset fails schema validation."""
    pass


class SchemaValidator:
    """Validator for verifying mandatory columns in customer loan datasets."""

    DEFAULT_REQUIRED_COLUMNS = [
        "ApplicantIncome",
        "LoanAmount",
        "Credit_History",
    ]

    def __init__(self, required_columns: Optional[List[str]] = None):
        self.required_columns = required_columns or self.DEFAULT_REQUIRED_COLUMNS

    def validate(self, df: pd.DataFrame) -> bool:
        """Check if all required columns exist in the DataFrame.

        Args:
            df: Target DataFrame to validate.

        Returns:
            bool: True if schema is valid.

        Raises:
            SchemaValidationError: If required columns are missing or input is invalid.
        """
        try:
            if not isinstance(df, pd.DataFrame):
                raise SchemaValidationError(f"Expected pandas DataFrame, got {type(df).__name__}")

            missing = [col for col in self.required_columns if col not in df.columns]
            if missing:
                raise SchemaValidationError(
                    f"Dataset schema validation failed. Missing required columns: {missing}"
                )
            return True
        except SchemaValidationError:
            raise
        except Exception as e:
            raise SchemaValidationError(f"Unexpected error during schema validation: {e}") from e
