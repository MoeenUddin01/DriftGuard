from pathlib import Path
from typing import Dict, Optional, Union
import logging
import time

from src.model.classifier import LoanClassifier
from src.model.trainer import ModelTrainer

logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """Top-level pipeline orchestrator for ML model training and artifact serialization."""

    def __init__(
        self,
        trainer: Optional[ModelTrainer] = None,
        model_type: str = "xgboost",
        target_col: str = "Loan_Status",
        **model_kwargs,
    ):
        """Initialize ModelTrainingPipeline.

        Args:
            trainer: ModelTrainer instance or None.
            model_type: Classifier model type ('xgboost', 'random_forest', 'gradient_boosting').
            target_col: Name of target column.
            **model_kwargs: Additional classifier hyperparameters.
        """
        if trainer is not None:
            self.trainer = trainer
        else:
            classifier = LoanClassifier(model_type=model_type, **model_kwargs)
            self.trainer = ModelTrainer(classifier=classifier, target_col=target_col)

    def run(
        self,
        processed_data_path: Union[str, Path] = "dataset/processed/train.parquet",
        artifact_dir: Union[str, Path] = "models",
    ) -> Dict[str, Union[LoanClassifier, Dict[str, Path], float]]:
        """Execute end-to-end model training pipeline.

        Args:
            processed_data_path: Path to input processed parquet or CSV training data.
            artifact_dir: Path to directory for saving model artifacts.

        Returns:
            Dict containing:
                - 'classifier': Fitted LoanClassifier instance
                - 'artifact_paths': Dict[str, Path] of saved artifact paths
                - 'execution_time_seconds': float execution duration
        """
        start_time = time.time()
        logger.info(f"Starting ModelTrainingPipeline execution using data at '{processed_data_path}'...")

        try:
            # 1. Train classifier from processed dataset file
            classifier = self.trainer.train_from_file(processed_data_path)

            # 2. Save serialized artifacts
            artifact_paths = self.trainer.save_artifacts(output_dir=artifact_dir)

            duration = round(time.time() - start_time, 4)
            logger.info(f"ModelTrainingPipeline completed successfully in {duration} seconds.")

            return {
                "classifier": classifier,
                "artifact_paths": artifact_paths,
                "execution_time_seconds": duration,
            }
        except Exception as e:
            logger.error(f"ModelTrainingPipeline execution failed: {e}")
            raise RuntimeError(f"ModelTrainingPipeline execution failed: {e}") from e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = ModelTrainingPipeline()
    try:
        results = pipeline.run()
        print(f"Pipeline finished! Artifacts saved: {results['artifact_paths']}")
    except Exception as err:
        print(f"Pipeline failed: {err}")
