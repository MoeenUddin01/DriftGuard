from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config.yaml")

_DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {"name": "DriftGuard", "version": "1.0.0"},
    "data": {
        "raw_path": "dataset/raw/loan_data.csv",
        "processed_dir": "dataset/processed/",
        "train_path": "dataset/processed/train.parquet",
        "val_path": "dataset/processed/val.parquet",
        "test_path": "dataset/processed/test.parquet",
        "baseline_stats_path": "dataset/baseline_stats.json",
    },
    "model": {
        "target_col": "Loan_Status",
        "artifact_path": "models/model.joblib",
        "random_state": 42,
        "test_size": 0.15,
        "val_size": 0.15,
        "epochs": 100,
    },
    "evaluation": {
        "min_roc_auc": 0.75,
    },
    "drift_detection": {
        "p_value_threshold": 0.05,
        "metrics_log_path": "logs/drift_metrics.log",
    },
}


def load_config(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration dictionary from YAML file with default fallback.

    Args:
        config_path: Path to config.yaml file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found at '{path}'. Falling back to internal defaults.")
        return _DEFAULT_CONFIG.copy()

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration successfully from '{path}'.")
            return config
    except Exception as e:
        logger.error(f"Failed to parse config file at '{path}': {e}. Using fallback defaults.")
        return _DEFAULT_CONFIG.copy()


def get_config_value(
    key_path: str,
    default: Any = None,
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
) -> Any:
    """Retrieve a nested configuration value by dot-separated key path.

    Example:
        get_config_value("model.epochs", 100) -> 100

    Args:
        key_path: Dot-separated string key path (e.g. "model.epochs").
        default: Fallback value if key is not found.
        config_path: Path to config.yaml file.

    Returns:
        Any: Configuration value or default.
    """
    config = load_config(config_path)
    keys = key_path.split(".")
    val: Any = config
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val
