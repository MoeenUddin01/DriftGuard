"""Alias module maintaining backwards compatibility for data_loader."""
from src.data.loader import DataLoader
from src.data.schema import SchemaValidationError, SchemaValidator
from src.data.metadata import MetadataExtractor

__all__ = ["DataLoader", "SchemaValidationError", "SchemaValidator", "MetadataExtractor"]
