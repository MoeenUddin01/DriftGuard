from typing import Any, Dict, Optional, Union
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb

logger = logging.getLogger(__name__)


class LoanClassifier:
    """Unified Loan Default Risk Binary Classifier supporting XGBoost and Scikit-Learn ensembles."""

    SUPPORTED_MODELS = {"xgboost", "random_forest", "gradient_boosting"}

    def __init__(
        self,
        model_type: str = "xgboost",
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        threshold: float = 0.5,
        early_stopping_rounds: Optional[int] = 10,
        epochs: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize LoanClassifier with algorithm type and hyperparameters.

        Args:
            model_type: Model framework name ('xgboost', 'random_forest', 'gradient_boosting').
            n_estimators: Number of boosting rounds / trees.
            max_depth: Maximum tree depth.
            learning_rate: Boosting learning rate (used for xgboost and gradient_boosting).
            random_state: Random seed for reproducibility.
            threshold: Probability threshold for classifying default risk (class 1).
            early_stopping_rounds: Rounds of no validation improvement before stopping early (XGBoost).
            epochs: Alias for n_estimators (number of boosting rounds/iterations).
            **kwargs: Additional model-specific hyperparameters.
        """
        model_type_lower = model_type.lower()
        if model_type_lower not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. Choose from: {sorted(self.SUPPORTED_MODELS)}"
            )

        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Probability threshold must be between 0.0 and 1.0, got {threshold}")

        self.model_type = model_type_lower
        self.epochs = epochs
        self.n_estimators = epochs if epochs is not None else n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.threshold = threshold
        self.early_stopping_rounds = early_stopping_rounds
        self.extra_kwargs = kwargs

        self.model = self._init_underlying_model()
        self.is_fitted: bool = False
        self.feature_names_: Optional[list] = None

    def _init_underlying_model(self) -> Any:
        """Instantiate underlying estimator model based on configuration."""
        if self.model_type == "xgboost":
            params = {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "random_state": self.random_state,
                "eval_metric": "logloss",
                **self.extra_kwargs,
            }
            return xgb.XGBClassifier(**params)

        elif self.model_type == "random_forest":
            params = {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "random_state": self.random_state,
                **self.extra_kwargs,
            }
            return RandomForestClassifier(**params)

        elif self.model_type == "gradient_boosting":
            params = {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "random_state": self.random_state,
                **self.extra_kwargs,
            }
            return GradientBoostingClassifier(**params)

        raise ValueError(f"Invalid model_type: {self.model_type}")

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[list] = None,
    ) -> "LoanClassifier":
        """Fit the loan classifier model on feature matrix X and target y.

        Args:
            X: Feature matrix (DataFrame or 2D NumPy array).
            y: Binary target vector (Series or 1D NumPy array).
            eval_set: Optional validation set list for early stopping [(X_val, y_val)].

        Returns:
            LoanClassifier: Fitted classifier instance.
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Input feature matrix X and target y must not be empty.")

        if len(X) != len(y):
            raise ValueError(f"Length mismatch between X ({len(X)}) and y ({len(y)}).")

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)

        try:
            logger.info(f"Fitting LoanClassifier ({self.model_type}) on {len(X)} samples...")
            if eval_set is not None and self.model_type == "xgboost":
                if self.early_stopping_rounds:
                    self.model.set_params(early_stopping_rounds=self.early_stopping_rounds)
                self.model.fit(X, y, eval_set=eval_set, verbose=False)
            else:
                self.model.fit(X, y)
            self.is_fitted = True
            logger.info(f"LoanClassifier ({self.model_type}) fitted successfully.")
            return self
        except Exception as e:
            logger.error(f"Failed to fit LoanClassifier: {e}")
            raise ValueError(f"LoanClassifier fitting failed: {e}") from e



    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities for X.

        Args:
            X: Feature matrix.

        Returns:
            np.ndarray: 2D array of shape (N, 2) with probabilities for [class 0, class 1].
        """
        if not self.is_fitted:
            raise RuntimeError("LoanClassifier must be fitted before calling predict_proba().")

        if len(X) == 0:
            raise ValueError("Input feature matrix X must not be empty.")

        try:
            proba = self.model.predict_proba(X)
            if proba.ndim == 1 or proba.shape[1] == 1:
                # Handle single-class output edge case cleanly
                p1 = proba.ravel()
                p0 = 1.0 - p1
                return np.column_stack([p0, p1])
            return proba
        except Exception as e:
            logger.error(f"Prediction probability computation failed: {e}")
            raise ValueError(f"Predict proba failed: {e}") from e

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict binary loan default risk classes for X based on configured threshold.

        Args:
            X: Feature matrix.

        Returns:
            np.ndarray: 1D array of binary predictions (0 = Non-Default, 1 = Default).
        """
        proba = self.predict_proba(X)
        default_proba = proba[:, 1]
        return (default_proba >= self.threshold).astype(int)

    def get_config(self) -> Dict[str, Any]:
        """Return model metadata configuration dictionary."""
        return {
            "model_type": self.model_type,
            "n_estimators": self.n_estimators,
            "epochs": self.epochs or self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "threshold": self.threshold,
            "extra_kwargs": self.extra_kwargs,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names_,
        }
