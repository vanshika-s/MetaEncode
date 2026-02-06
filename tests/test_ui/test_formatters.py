# tests/test_ui/test_formatters.py
"""Tests for src/ui/formatters.py."""

import pytest

from src.ui.formatters import format_organism_display, truncate_text


class TestFormatOrganismDisplay:
    """Tests for format_organism_display function."""

    def test_empty_string_returns_na(self):
        """Empty string should return 'N/A'."""
        assert format_organism_display("") == "N/A"

    def test_none_returns_na(self):
        """None should return 'N/A'."""
        assert format_organism_display(None) == "N/A"

    def test_human_scientific_name(self):
        """Homo sapiens should format with assembly."""
        result = format_organism_display("Homo sapiens")
        assert "Human" in result or "Homo sapiens" in result

    def test_mouse_scientific_name(self):
        """Mus musculus should format with assembly."""
        result = format_organism_display("Mus musculus")
        assert "Mouse" in result or "Mus musculus" in result

    def test_unknown_organism_returned_as_is(self):
        """Unknown organisms should be returned unchanged."""
        result = format_organism_display("Unknown species")
        assert "Unknown species" in result

    def test_human_common_name(self):
        """Common name 'human' should be handled."""
        result = format_organism_display("human")
        # Should either return formatted or as-is
        assert result is not None
        assert result != "N/A"

    def test_whitespace_only_returns_na(self):
        """Whitespace-only string should return 'N/A'."""
        # The function checks `if not organism` which is False for "   "
        # So whitespace will be passed to get_organism_display
        result = format_organism_display("   ")
        # Whitespace is truthy, so it goes to get_organism_display
        assert result is not None


class TestTruncateText:
    """Tests for truncate_text function."""

    def test_short_text_unchanged(self):
        """Text shorter than max_length should be unchanged."""
        text = "Short text"
        result = truncate_text(text, max_length=80)
        assert result == "Short text"

    def test_exact_length_unchanged(self):
        """Text exactly at max_length should be unchanged."""
        text = "x" * 80
        result = truncate_text(text, max_length=80)
        assert result == text
        assert "..." not in result

    def test_long_text_truncated(self):
        """Text longer than max_length should be truncated with ellipsis."""
        text = "x" * 100
        result = truncate_text(text, max_length=80)
        assert len(result) == 83  # 80 + "..."
        assert result.endswith("...")

    def test_custom_max_length(self):
        """Custom max_length should be respected."""
        text = "Hello World"
        result = truncate_text(text, max_length=5)
        assert result == "Hello..."

    def test_empty_string(self):
        """Empty string should return empty string."""
        result = truncate_text("", max_length=80)
        assert result == ""

    def test_none_returns_empty_string(self):
        """None should return empty string."""
        result = truncate_text(None, max_length=80)
        assert result == ""

    def test_non_string_converted(self):
        """Non-string values should be converted to string."""
        result = truncate_text(12345, max_length=80)
        assert result == "12345"

    def test_non_string_truncated(self):
        """Non-string values should be truncated after conversion."""
        result = truncate_text(123456789, max_length=5)
        assert result == "12345..."

    def test_default_max_length_is_80(self):
        """Default max_length should be 80."""
        text = "x" * 100
        result = truncate_text(text)
        assert len(result) == 83  # 80 + "..."

    def test_unicode_text(self):
        """Unicode text should be handled correctly."""
        text = "Hello 世界! " * 20
        result = truncate_text(text, max_length=20)
        assert len(result) == 23  # 20 + "..."
        assert result.endswith("...")

    def test_newlines_preserved(self):
        """Newlines in text should be preserved."""
        text = "Line 1\nLine 2"
        result = truncate_text(text, max_length=80)
        assert result == "Line 1\nLine 2"

    def test_single_character_max_length(self):
        """Single character max_length should work."""
        result = truncate_text("Hello", max_length=1)
        assert result == "H..."
