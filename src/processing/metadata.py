# src/processing/metadata.py
"""Metadata extraction and cleaning utilities.

This module handles the preprocessing of ENCODE experiment metadata,
including text normalization, missing value handling, and field extraction.
"""

import re
from typing import Any, Optional

import pandas as pd

from src.ui.vocabularies import get_primary_organ_for_biosample


class MetadataProcessor:
    """Process and clean ENCODE experiment metadata.

    This class handles:
    - Text field normalization (lowercase, special char removal)
    - Missing value imputation
    - Field extraction from nested JSON structures
    - Metadata validation

    Example:
        >>> processor = MetadataProcessor()
        >>> clean_df = processor.process(raw_experiments_df)
    """

    # Fields to extract and process
    TEXT_FIELDS = ["description", "title"]
    CATEGORICAL_FIELDS = [
        "assay_term_name",
        "organism",
        "biosample_term_name",
        "lab",
        "life_stage",
    ]
    NUMERIC_FIELDS = ["replicate_count", "file_count"]

    # Required fields for a valid record
    REQUIRED_FIELDS = ["accession"]
    # At least one of these must be present
    REQUIRED_ONE_OF = ["description", "title", "assay_term_name"]

    def __init__(self, fill_missing: bool = True) -> None:
        """Initialize the metadata processor.

        Args:
            fill_missing: Whether to fill missing values with defaults.
        """
        self.fill_missing = fill_missing

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw experiment metadata DataFrame.

        Args:
            df: Raw DataFrame from ENCODE API.

        Returns:
            Processed DataFrame with cleaned and normalized fields.
        """
        if df.empty:
            return df.copy()

        result = df.copy()

        # Clean text fields
        for field in self.TEXT_FIELDS:
            if field in result.columns:
                result[f"{field}_clean"] = result[field].apply(self.clean_text)

        # Create combined text field for embedding
        text_cols = [
            f"{f}_clean" for f in self.TEXT_FIELDS if f"{f}_clean" in result.columns
        ]
        if text_cols:
            result["combined_text"] = result[text_cols].apply(
                lambda row: " ".join(str(v) for v in row if v), axis=1
            )

        # Fill missing values if requested
        if self.fill_missing:
            # Fill text fields with empty string
            for field in self.TEXT_FIELDS:
                if field in result.columns:
                    result[field] = result[field].fillna("")
                clean_field = f"{field}_clean"
                if clean_field in result.columns:
                    result[clean_field] = result[clean_field].fillna("")

            # Fill categorical fields with "unknown"
            for field in self.CATEGORICAL_FIELDS:
                if field in result.columns:
                    result[field] = result[field].fillna("unknown")

            # Fill numeric fields with 0
            for field in self.NUMERIC_FIELDS:
                if field in result.columns:
                    result[field] = pd.to_numeric(
                        result[field], errors="coerce"
                    ).fillna(0)

            # Fill combined_text if it exists
            if "combined_text" in result.columns:
                result["combined_text"] = result["combined_text"].fillna("")

        # Add organ column derived from biosample_term_name
        if "biosample_term_name" in result.columns:
            result["organ"] = (
                result["biosample_term_name"]
                .apply(
                    lambda x: (
                        get_primary_organ_for_biosample(x)
                        if pd.notna(x) and x != "unknown"
                        else None
                    )
                )
                .fillna("unknown")
            )

        return result

    def clean_text(self, text: Optional[str]) -> str:
        """Clean and normalize text field.

        Applies the following transformations:
        - Convert to lowercase
        - Remove special characters (keep alphanumeric and whitespace)
        - Normalize whitespace (collapse multiple spaces to one)
        - Strip leading/trailing whitespace

        Args:
            text: Raw text string (may be None).

        Returns:
            Cleaned text (lowercase, stripped, special chars removed).
        """
        if text is None or pd.isna(text):
            return ""

        text = str(text)

        # Convert to lowercase
        text = text.lower()

        # Remove special characters, keep alphanumeric and whitespace
        text = re.sub(r"[^\w\s]", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def extract_nested_field(self, data: dict, field_path: str) -> Optional[Any]:
        """Extract a value from nested dictionary using dot notation.

        Args:
            data: Dictionary (possibly nested).
            field_path: Dot-separated path (e.g., "biosample_ontology.term_name").

        Returns:
            Extracted value or None if not found.
        """
        if not data or not field_path:
            return None

        keys = field_path.split(".")
        current: Any = data

        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
                if current is None:
                    return None
            else:
                return None

        return current

    def validate_record(self, record: dict) -> bool:
        """Validate that a record has minimum required metadata.

        A record is valid if it has:
        - All required fields (accession)
        - At least one of: description, title, or assay_term_name

        Args:
            record: Dictionary containing experiment metadata.

        Returns:
            True if record has required fields, False otherwise.
        """
        if not record:
            return False

        # Check all required fields are present and non-empty
        for field in self.REQUIRED_FIELDS:
            value = record.get(field)
            if not value or (isinstance(value, str) and not value.strip()):
                return False

        # Check at least one of the optional required fields is present
        has_one_of = False
        for field in self.REQUIRED_ONE_OF:
            value = record.get(field)
            if value and (not isinstance(value, str) or value.strip()):
                has_one_of = True
                break

        return has_one_of
