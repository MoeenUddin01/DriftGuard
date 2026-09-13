import logging
from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates binary classification model performance."""

    def __init__(self, target_col: str = "Loan_Status"):
        self.target_col = target_col

    def _extract_x_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Extract feature matrix X and target y from DataFrame."""
        if self.target_col not in df.columns:
            # Try case-insensitive matching
            target_lower = self.target_col.lower()
            matched = False
            for col in df.columns:
                if col.lower() == target_lower:
                    self.target_col = col
                    matched = True
                    break
            if not matched:
                raise ValueError(
                    f"Target column '{self.target_col}' not found in dataset columns: {list(df.columns)}"
                )

        y = df[self.target_col].copy()
        
        # Map string labels if necessary
        if pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y):
            y = y.astype(str).str.strip().map({"Y": 1, "N": 0, "1": 1, "0": 0})
            if y.isnull().any():
                raise ValueError("Target column contains unrecognized string labels.")

        y_vals = y.values.astype(int)
        
        # Drop target and any potential ID column
        drop_cols = [c for c in [self.target_col, "Loan_ID"] if c in df.columns]
        X = df.drop(columns=drop_cols)
        
        return X, y_vals

    def evaluate(self, model: Any, df: pd.DataFrame) -> Dict[str, Union[float, Dict[str, int]]]:
        """Calculate classification metrics against a test DataFrame.
        
        Args:
            model: Fitted LoanClassifier or similar object with predict() and predict_proba().
            df: Test DataFrame containing features and target column.
            
        Returns:
            Dictionary containing ROC-AUC, F1, Precision, Recall, Accuracy, and Confusion Matrix.
        """
        logger.info(f"Evaluating model on {len(df)} samples...")
        
        X_test, y_true = self._extract_x_y(df)
        
        if len(X_test) == 0:
            raise ValueError("Test feature matrix is empty after dropping target columns.")
            
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        metrics = {}
        
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            metrics["f1_score"] = float(f1_score(y_true, y_pred))
            metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
            metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics["confusion_matrix"] = {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp)
                }
            else:
                metrics["confusion_matrix"] = {"raw": cm.tolist()}
                
            logger.info(f"Evaluation complete. ROC-AUC: {metrics['roc_auc']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate evaluation metrics: {e}")
            raise ValueError(f"Metric calculation failed: {e}") from e

    def validate_threshold(self, metrics: Dict[str, Any], min_roc_auc: float = 0.75) -> bool:
        """Validate if model meets minimum performance threshold for production deployment.
        
        Args:
            metrics: Dictionary of calculated metrics from evaluate().
            min_roc_auc: Minimum acceptable ROC-AUC score.
            
        Returns:
            bool: True if metrics meet threshold, False otherwise.
        """
        roc_auc = metrics.get("roc_auc", 0.0)
        is_valid = roc_auc >= min_roc_auc
        
        if is_valid:
            logger.info(f"Model PASSES validation threshold (ROC-AUC {roc_auc:.4f} >= {min_roc_auc:.4f})")
        else:
            logger.warning(f"Model FAILS validation threshold (ROC-AUC {roc_auc:.4f} < {min_roc_auc:.4f})")
            
        return is_valid
