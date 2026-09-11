from pathlib import Path
from typing import Any, Dict, Optional, Union
import json
import logging
import joblib
import pandas as pd

from src.model.classifier import LoanClassifier

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Model training orchestrator and artifact serializer for Loan Default Risk prediction."""

    def __init__(
        self,
        classifier: Optional[LoanClassifier] = None,
        target_col: str = "Loan_Status",
        id_col: str = "Loan_ID",
    ):
        """Initialize ModelTrainer.

        Args:
            classifier: Pre-configured LoanClassifier instance or None (uses default XGBoost).
            target_col: Name of target label column in dataset.
            id_col: Name of identifier column to drop before training if present.
        """
        self.classifier = classifier or LoanClassifier()
        self.target_col = target_col
        self.id_col = id_col

    def _resolve_target_column(self, df: pd.DataFrame) -> str:
        """Resolve case variations for target column in DataFrame."""
        if self.target_col in df.columns:
            return self.target_col
        
        # Check case-insensitive match
        target_lower = self.target_col.lower()
        for col in df.columns:
            if col.lower() == target_lower:
                return col

        raise ValueError(
            f"Target column '{self.target_col}' not found in dataset columns: {list(df.columns)}"
        )

    def _extract_x_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        """Extract feature matrix X and binary target vector y from DataFrame."""
        target_name = self._resolve_target_column(df)
        y = df[target_name].values
        if pd.api.types.is_object_dtype(df[target_name]) or pd.api.types.is_string_dtype(df[target_name]):
            y = pd.Series(df[target_name]).astype(str).str.strip().map({"Y": 1, "N": 0, "1": 1, "0": 0}).values

        drop_cols = [c for c in [target_name, self.id_col] if c in df.columns]
        X = df.drop(columns=drop_cols)
        if X.shape[1] == 0:
            raise ValueError("No feature columns remaining after dropping target and ID columns.")
        return X, y

    def train_from_dataframe(
        self, df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None
    ) -> LoanClassifier:
        """Fit classifier using an in-memory DataFrame with optional validation set.

        Args:
            df: Processed DataFrame containing features and target column.
            val_df: Optional validation DataFrame for evaluation during training.

        Returns:
            LoanClassifier: Fitted classifier instance.
        """
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            raise ValueError("Input DataFrame for training is empty or invalid.")

        X_train, y_train = self._extract_x_y(df)
        eval_set = None

        if val_df is not None and len(val_df) > 0:
            X_val, y_val = self._extract_x_y(val_df)
            eval_set = [(X_val, y_val)]
            logger.info(f"Using validation set of shape X_val={X_val.shape} for training evaluation.")

        logger.info(f"Training model on DataFrame with shape X={X_train.shape}, y={y_train.shape}...")
        self.classifier.fit(X_train, y_train, eval_set=eval_set)
        return self.classifier

    def train_from_file(
        self, data_path: Union[str, Path], val_data_path: Optional[Union[str, Path]] = None
    ) -> LoanClassifier:
        """Load parquet or CSV processed dataset from disk and fit classifier.

        Args:
            data_path: Path to dataset file (`.parquet` or `.csv`).
            val_data_path: Optional path to validation dataset file (`val.parquet`).

        Returns:
            LoanClassifier: Fitted classifier instance.
        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Training dataset file not found at: '{path}'")

        try:
            logger.info(f"Loading training data from '{path}'...")
            if path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path)
            elif path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            else:
                raise ValueError(f"Unsupported file format '{path.suffix}'. Use .parquet or .csv.")
        except Exception as e:
            logger.error(f"Failed to read training data from '{path}': {e}")
            raise IOError(f"Error reading dataset at '{path}': {e}") from e

        val_df = None
        if val_data_path is not None and Path(val_data_path).exists():
            val_path = Path(val_data_path)
            logger.info(f"Loading validation data from '{val_path}'...")
            val_df = pd.read_parquet(val_path) if val_path.suffix.lower() == ".parquet" else pd.read_csv(val_path)

        return self.train_from_dataframe(df, val_df=val_df)


    def save_artifacts(
        self,
        output_dir: Union[str, Path],
        model_filename: str = "model.joblib",
        config_filename: str = "training_config.json",
    ) -> Dict[str, Path]:
        """Serialize trained model artifact and configuration metadata.

        Args:
            output_dir: Directory path for saving artifacts.
            model_filename: Filename for serialized model artifact.
            config_filename: Filename for JSON metadata.

        Returns:
            Dict[str, Path]: Dictionary mapping artifact key to saved absolute path.
        """
        if not self.classifier.is_fitted:
            raise RuntimeError("Cannot save artifacts because the classifier has not been fitted.")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        model_file = out_path / model_filename
        config_file = out_path / config_filename

        try:
            logger.info(f"Saving serialized model artifact to '{model_file}'...")
            joblib.dump(self.classifier, model_file)

            config_data = self.classifier.get_config()
            config_data["target_column"] = self.target_col
            config_data["model_artifact_path"] = str(model_file.resolve())

            logger.info(f"Saving training config metadata to '{config_file}'...")
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2)

            return {
                "model_path": model_file,
                "config_path": config_file,
            }
        except Exception as e:
            logger.error(f"Failed to save training artifacts to '{output_dir}': {e}")
            raise IOError(f"Error saving model artifacts to '{output_dir}': {e}") from e
