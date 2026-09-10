"""Data ingestion and validation module."""
from src.data.data_loader import DataLoader, SchemaValidationError

__all__ = ["DataLoader", "SchemaValidationError"]
