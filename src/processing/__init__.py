# src/processing/__init__.py
"""Metadata processing and encoding module."""

from .encoders import CategoricalEncoder, NumericEncoder
from .metadata import MetadataProcessor

__all__ = ["MetadataProcessor", "CategoricalEncoder", "NumericEncoder"]
