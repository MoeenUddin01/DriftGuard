"""Alias module maintaining backwards compatibility for data_preprocessing."""
from src.pipeline.preprocessor import DataPreprocessor
from src.pipeline.partition import save_processed_data, DatasetPartitioner
from src.pipeline.data_pipeline import DataIngestionPipeline

__all__ = [
    "DataPreprocessor",
    "save_processed_data",
    "DatasetPartitioner",
    "DataIngestionPipeline",
]
