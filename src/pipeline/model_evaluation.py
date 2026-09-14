from pathlib import Path
from typing import Dict, Any, Union, Optional
import json
import logging
import joblib
import pandas as pd
import numpy as np

from src.model.evaluation import ModelEvaluator
from src.utils.config import get_config_value

logger = logging.getLogger(__name__)


def log_baseline_stats(df: pd.DataFrame, save_path: Union[str, Path]) -> Dict[str, Any]:
    """Compute and serialize baseline distribution statistics for features.
    
    Computes mean, std, percentiles for numerical columns and frequencies for categoricals.
    
    Args:
        df: Training/Validation dataset used to compute baseline distributions.
        save_path: File path to save baseline_stats.json.
        
    Returns:
        Dictionary containing the baseline statistical profiles.
    """
    logger.info(f"Computing baseline statistics on {len(df)} samples...")
    stats: Dict[str, Any] = {"num_samples": len(df), "features": {}}
    
    for col in df.columns:
        # Skip target and ID columns if present
        if col in ["Loan_Status", "Loan_ID"]:
            continue
            
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
            
        if pd.api.types.is_numeric_dtype(col_data):
            stats["features"][col] = {
                "type": "numerical",
                "mean": float(col_data.mean()),
                "std": float(col_data.std()) if len(col_data) > 1 else 0.0,
                "min": float(col_data.min()),
                "25%": float(np.percentile(col_data, 25)),
                "50%": float(np.percentile(col_data, 50)),
                "75%": float(np.percentile(col_data, 75)),
                "max": float(col_data.max()),
                "missing_count": int(df[col].isnull().sum()),
                "deciles": [float(np.percentile(col_data, p)) for p in range(0, 101, 10)],
                "quantiles_100": [float(np.percentile(col_data, p)) for p in range(0, 101, 1)],
            }
        else:
            value_counts = col_data.value_counts(normalize=True).to_dict()
            # Convert counts to standard Python types
            clean_counts = {str(k): float(v) for k, v in value_counts.items()}
            stats["features"][col] = {
                "type": "categorical",
                "frequencies": clean_counts,
                "missing_count": int(df[col].isnull().sum()),
                "unique_categories": len(clean_counts),
            }
            
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Baseline statistics saved successfully to {out_path}")
        return stats
    except Exception as e:
        logger.error(f"Failed to save baseline statistics: {e}")
        raise IOError(f"Error writing baseline stats to {out_path}: {e}") from e


class ModelEvaluationPipeline:
    """End-to-End Orchestrator for evaluating models and logging baseline stats."""
    
    def __init__(self, evaluator: Optional[ModelEvaluator] = None):
        self.evaluator = evaluator or ModelEvaluator()
        
    def run(
        self,
        model_path: Optional[Union[str, Path]] = None,
        test_data_path: Optional[Union[str, Path]] = None,
        train_data_path: Optional[Union[str, Path]] = None,
        baseline_save_path: Optional[Union[str, Path]] = None,
        min_roc_auc: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute evaluation, validation, and baseline logging.
        
        Args:
            model_path: Path to serialized model artifact.
            test_data_path: Path to unseen test dataset for evaluation.
            train_data_path: Path to training dataset to compute baselines.
            baseline_save_path: Path to save JSON baseline stats.
            min_roc_auc: Minimum ROC-AUC score required to pass validation.
            
        Returns:
            Dict with execution results and metrics.
        """
        resolved_model_path = model_path or get_config_value("model.artifact_path", "models/model.joblib")
        resolved_test_data_path = test_data_path or get_config_value("data.test_path", "dataset/processed/test.parquet")
        resolved_train_data_path = train_data_path or get_config_value("data.train_path", "dataset/processed/train.parquet")
        resolved_baseline_save_path = baseline_save_path or get_config_value("data.baseline_stats_path", "dataset/baseline_stats.json")
        resolved_min_roc_auc = min_roc_auc if min_roc_auc is not None else get_config_value("evaluation.min_roc_auc", 0.75)

        logger.info("Starting ModelEvaluationPipeline...")
        
        # 1. Load Artifacts & Data
        try:
            model = joblib.load(resolved_model_path)
            
            test_path = Path(resolved_test_data_path)
            test_df = pd.read_parquet(test_path) if test_path.suffix == ".parquet" else pd.read_csv(test_path)
        except Exception as e:
            logger.error(f"Failed to load model or test data: {e}")
            raise RuntimeError(f"Pipeline failed during artifact loading: {e}") from e
            
        # 2. Evaluate Model
        metrics = self.evaluator.evaluate(model, test_df)
        
        # 3. Validate Threshold
        is_valid = self.evaluator.validate_threshold(metrics, min_roc_auc=resolved_min_roc_auc)
        
        results = {
            "metrics": metrics,
            "is_valid": is_valid,
            "threshold_used": resolved_min_roc_auc,
        }
        
        # 4. Log Baselines if model is valid
        if is_valid:
            logger.info("Model validated successfully. Computing baseline distributions...")
            try:
                train_path = Path(resolved_train_data_path)
                train_df = pd.read_parquet(train_path) if train_path.suffix == ".parquet" else pd.read_csv(train_path)
                
                baseline_stats = log_baseline_stats(train_df, resolved_baseline_save_path)
                results["baseline_stats_path"] = str(Path(resolved_baseline_save_path).resolve())
                
                # Mock "deployment" symlink update (could be implemented fully in future)
                logger.info(f"Deployment Promotion: Model at {model_path} meets standards.")
            except Exception as e:
                logger.error(f"Failed to log baseline stats: {e}")
                raise RuntimeError(f"Pipeline failed during baseline logging: {e}") from e
        else:
            logger.warning(f"Deployment Aborted: Model failed validation threshold {min_roc_auc}.")
            
        return results
