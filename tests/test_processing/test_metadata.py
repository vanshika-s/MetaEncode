# tests/test_processing/test_metadata.py
"""Tests for metadata processing utilities."""

import pandas as pd

from src.processing.metadata import MetadataProcessor


class TestMetadataProcessor:
    """Test suite for MetadataProcessor."""

    def test_processor_initialization(self):
        """Test that processor initializes correctly."""
        processor = MetadataProcessor()
        assert processor.fill_missing is True

        processor_no_fill = MetadataProcessor(fill_missing=False)
        assert processor_no_fill.fill_missing is False

    def test_clean_text_removes_special_chars(self):
        """Test that clean_text normalizes text properly."""
        processor = MetadataProcessor()
        result = processor.clean_text("ChIP-seq on K562 (human)")
        # Special chars become spaces, multiple spaces collapse
        assert result == "chip seq on k562 human"

    def test_clean_text_handles_none(self):
        """Test that clean_text handles None input."""
        processor = MetadataProcessor()
        result = processor.clean_text(None)
        assert result == ""

    def test_clean_text_handles_empty_string(self):
        """Test that clean_text handles empty string."""
        processor = MetadataProcessor()
        result = processor.clean_text("")
        assert result == ""

    def test_clean_text_normalizes_whitespace(self):
        """Test that clean_text normalizes whitespace."""
        processor = MetadataProcessor()
        result = processor.clean_text("  multiple   spaces  ")
        assert result == "multiple spaces"

    def test_extract_nested_field(self, sample_experiment_data):
        """Test extraction of nested fields using dot notation."""
        processor = MetadataProcessor()
        result = processor.extract_nested_field(
            sample_experiment_data, "biosample_ontology.term_name"
        )
        assert result == "K562"

    def test_extract_nested_field_nonexistent(self, sample_experiment_data):
        """Test extraction of non-existent nested field returns None."""
        processor = MetadataProcessor()
        result = processor.extract_nested_field(
            sample_experiment_data, "nonexistent.field"
        )
        assert result is None

    def test_extract_nested_field_empty_path(self, sample_experiment_data):
        """Test extraction with empty path returns None."""
        processor = MetadataProcessor()
        result = processor.extract_nested_field(sample_experiment_data, "")
        assert result is None

    def test_validate_record_with_valid_data(self, sample_experiment_data):
        """Test validation passes for record with required fields."""
        processor = MetadataProcessor()
        assert processor.validate_record(sample_experiment_data) is True

    def test_validate_record_with_missing_fields(self):
        """Test validation fails for record missing required fields."""
        processor = MetadataProcessor()
        # Has accession but missing description/title/assay
        incomplete_record = {"accession": "ENCSR000AAA"}
        assert processor.validate_record(incomplete_record) is False

    def test_validate_record_with_empty_accession(self):
        """Test validation fails for record with empty accession."""
        processor = MetadataProcessor()
        record = {"accession": "", "description": "Some description"}
        assert processor.validate_record(record) is False

    def test_validate_record_with_none(self):
        """Test validation fails for None record."""
        processor = MetadataProcessor()
        assert processor.validate_record(None) is False

    def test_process_creates_combined_text(self, sample_experiments_df):
        """Test that process creates combined_text column."""
        processor = MetadataProcessor()
        result = processor.process(sample_experiments_df)
        assert "combined_text" in result.columns
        assert "description_clean" in result.columns

    def test_process_fills_missing_values(self):
        """Test that process fills missing values when fill_missing=True."""
        processor = MetadataProcessor(fill_missing=True)
        df = pd.DataFrame(
            {
                "accession": ["ENCSR000AAA"],
                "description": [None],
                "assay_term_name": [None],
                "replicate_count": [None],
            }
        )
        result = processor.process(df)
        assert result["description"].iloc[0] == ""
        assert result["assay_term_name"].iloc[0] == "unknown"

    def test_process_empty_dataframe(self):
        """Test that process handles empty DataFrame."""
        processor = MetadataProcessor()
        df = pd.DataFrame()
        result = processor.process(df)
        assert result.empty


# ============================================================================
# Additional Edge Case Tests for Coverage
# ============================================================================


class TestMetadataProcessorCoverageEdgeCases:
    """Additional edge case tests for MetadataProcessor coverage."""

    def test_extract_nested_non_dict_intermediate(self):
        """Line 163: Nested path hits non-dict intermediate value."""
        processor = MetadataProcessor()
        # Path "a.b.c" where "a.b" is a string (not a dict)
        data = {"a": {"b": "string_value"}}
        result = processor.extract_nested_field(data, "a.b.c")
        # Should return None because "string_value" is not a dict
        assert result is None

    def test_extract_nested_list_intermediate(self):
        """Line 163: Nested path hits list intermediate value."""
        processor = MetadataProcessor()
        # Path "a.b.c" where "a.b" is a list (not a dict)
        data = {"a": {"b": [1, 2, 3]}}
        result = processor.extract_nested_field(data, "a.b.c")
        # Should return None because [1, 2, 3] is not a dict
        assert result is None
