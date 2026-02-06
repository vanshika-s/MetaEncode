# tests/test_ui/test_handlers.py
"""Tests for src/ui/handlers.py."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ui.handlers import apply_spell_correction, execute_search
from src.ui.search_filters import FilterState


class TestApplySpellCorrection:
    """Tests for apply_spell_correction function."""

    def test_empty_string_returns_empty(self):
        """Empty string should return empty string and no message."""
        corrected, msg = apply_spell_correction("")
        assert corrected == ""
        assert msg is None

    def test_none_returns_empty(self):
        """None should return empty string and no message."""
        corrected, msg = apply_spell_correction(None)
        assert corrected == ""
        assert msg is None

    @patch("src.ui.handlers.correct_spelling")
    def test_no_corrections_needed(self, mock_correct):
        """When no corrections are made, message should be None."""
        # Mock returns same word (no correction)
        mock_correct.side_effect = lambda x: x

        corrected, msg = apply_spell_correction("hello world")

        assert corrected == "hello world"
        assert msg is None

    @patch("src.ui.handlers.correct_spelling")
    def test_single_word_corrected(self, mock_correct):
        """Single word correction should be reported."""
        mock_correct.side_effect = lambda x: "hello" if x == "helo" else x

        corrected, msg = apply_spell_correction("helo world")

        assert corrected == "hello world"
        assert msg is not None
        assert "helo -> hello" in msg

    @patch("src.ui.handlers.correct_spelling")
    def test_multiple_words_corrected(self, mock_correct):
        """Multiple word corrections should all be reported."""

        def correct(word):
            corrections = {"helo": "hello", "wrld": "world"}
            return corrections.get(word, word)

        mock_correct.side_effect = correct

        corrected, msg = apply_spell_correction("helo wrld")

        assert corrected == "hello world"
        assert msg is not None
        assert "helo -> hello" in msg
        assert "wrld -> world" in msg

    @patch("src.ui.handlers.correct_spelling")
    def test_case_insensitive_comparison(self, mock_correct):
        """Correction comparison should be case-insensitive."""
        # Return same word with different case - should not count as correction
        mock_correct.side_effect = lambda x: x.capitalize()

        corrected, msg = apply_spell_correction("hello")

        # "hello" -> "Hello" (same word, different case)
        assert corrected == "Hello"
        assert msg is None  # No correction message since it's the same word

    @patch("src.ui.handlers.correct_spelling")
    def test_preserves_word_order(self, mock_correct):
        """Word order should be preserved after correction."""
        mock_correct.side_effect = lambda x: x

        corrected, msg = apply_spell_correction("one two three four")

        assert corrected == "one two three four"

    @patch("src.ui.handlers.correct_spelling")
    def test_handles_single_word(self, mock_correct):
        """Single word input should be handled."""
        mock_correct.return_value = "corrected"

        corrected, msg = apply_spell_correction("word")

        assert corrected == "corrected"


class TestExecuteSearch:
    """Tests for execute_search function."""

    @patch("src.ui.handlers.get_filter_manager")
    @patch("src.ui.handlers.get_api_client")
    def test_basic_search_returns_dataframe(self, mock_client, mock_filter_mgr):
        """Basic search should return a DataFrame."""
        # Setup mock client
        client = MagicMock()
        client.fetch_experiments.return_value = pd.DataFrame(
            {"accession": ["ENCSR001", "ENCSR002"], "assay_term_name": ["ChIP-seq", "RNA-seq"]}
        )
        mock_client.return_value = client

        # Setup mock filter manager
        filter_mgr = MagicMock()
        mock_filter_mgr.return_value = filter_mgr

        filter_state = FilterState(assay_type="ChIP-seq")
        results, msg = execute_search(filter_state, max_results=10)

        assert isinstance(results, pd.DataFrame)
        assert len(results) <= 10

    @patch("src.ui.handlers.get_filter_manager")
    @patch("src.ui.handlers.get_api_client")
    def test_organism_mapping_human(self, mock_client, mock_filter_mgr):
        """Human organism should be mapped to scientific name."""
        client = MagicMock()
        client.fetch_experiments.return_value = pd.DataFrame()
        mock_client.return_value = client
        mock_filter_mgr.return_value = MagicMock()

        filter_state = FilterState(organism="human")
        execute_search(filter_state, max_results=10)

        # Check that fetch_experiments was called with scientific name
        call_kwargs = client.fetch_experiments.call_args.kwargs
        assert call_kwargs["organism"] == "Homo sapiens"

    @patch("src.ui.handlers.get_filter_manager")
    @patch("src.ui.handlers.get_api_client")
    def test_organism_mapping_mouse(self, mock_client, mock_filter_mgr):
        """Mouse organism should be mapped to scientific name."""
        client = MagicMock()
        client.fetch_experiments.return_value = pd.DataFrame()
        mock_client.return_value = client
        mock_filter_mgr.return_value = MagicMock()

        filter_state = FilterState(organism="mouse")
        execute_search(filter_state, max_results=10)

        call_kwargs = client.fetch_experiments.call_args.kwargs
        assert call_kwargs["organism"] == "Mus musculus"

    @patch("src.ui.handlers.get_filter_manager")
    @patch("src.ui.handlers.get_api_client")
    def test_unknown_organism_passed_through(self, mock_client, mock_filter_mgr):
        """Unknown organisms should be passed through unchanged."""
        client = MagicMock()
        client.fetch_experiments.return_value = pd.DataFrame()
        mock_client.return_value = client
        mock_filter_mgr.return_value = MagicMock()

        filter_state = FilterState(organism="Danio rerio")
        execute_search(filter_state, max_results=10)

        call_kwargs = client.fetch_experiments.call_args.kwargs
        assert call_kwargs["organism"] == "Danio rerio"

    @patch("src.ui.handlers.get_filter_manager")
    @patch("src.ui.handlers.get_api_client")
    @patch("src.ui.handlers.apply_spell_correction")
    def test_spell_correction_applied(self, mock_spell, mock_client, mock_filter_mgr):
        """Spell correction should be applied to description search."""
        mock_spell.return_value = ("corrected term", "Corrected: term -> corrected")

        client = MagicMock()
        client.fetch_experiments.return_value = pd.DataFrame()
        mock_client.return_value = client
        mock_filter_mgr.return_value = MagicMock()

        filter_state = FilterState(description_search="misspeled")
        results, msg = execute_search(filter_state, max_results=10)

        mock_spell.assert_called_once_with("misspeled")
        assert msg == "Corrected: term -> corrected"

    @patch("src.ui.handlers.get_filter_manager")
    @patch("src.ui.handlers.get_api_client")
    def test_post_filtering_applied(self, mock_client, mock_filter_mgr):
        """Post-filtering should be applied when results are not empty."""
        client = MagicMock()
        client.fetch_experiments.return_value = pd.DataFrame(
            {"accession": ["ENCSR001", "ENCSR002", "ENCSR003"]}
        )
        mock_client.return_value = client

        filter_mgr = MagicMock()
        filter_mgr.apply_filters.return_value = pd.DataFrame(
            {"accession": ["ENCSR001"]}
        )
        mock_filter_mgr.return_value = filter_mgr

        filter_state = FilterState(assay_type="ChIP-seq", lab="Some Lab")
        results, _ = execute_search(filter_state, max_results=10)

        # Filter manager should have been called for post-filtering
        filter_mgr.apply_filters.assert_called_once()

    @patch("src.ui.handlers.get_filter_manager")
    @patch("src.ui.handlers.get_api_client")
    def test_empty_results_not_filtered(self, mock_client, mock_filter_mgr):
        """Empty results should not trigger post-filtering."""
        client = MagicMock()
        client.fetch_experiments.return_value = pd.DataFrame()
        mock_client.return_value = client

        filter_mgr = MagicMock()
        mock_filter_mgr.return_value = filter_mgr

        filter_state = FilterState(assay_type="ChIP-seq", lab="Some Lab")
        results, _ = execute_search(filter_state, max_results=10)

        # Filter manager apply_filters should not be called for empty results
        filter_mgr.apply_filters.assert_not_called()

    @patch("src.ui.handlers.get_filter_manager")
    @patch("src.ui.handlers.get_api_client")
    def test_results_limited_to_max_results(self, mock_client, mock_filter_mgr):
        """Results should be limited to max_results."""
        client = MagicMock()
        client.fetch_experiments.return_value = pd.DataFrame(
            {"accession": [f"ENCSR{i:03d}" for i in range(100)]}
        )
        mock_client.return_value = client
        mock_filter_mgr.return_value = MagicMock()

        filter_state = FilterState(assay_type="ChIP-seq")
        results, _ = execute_search(filter_state, max_results=5)

        assert len(results) == 5

    @patch("src.ui.handlers.get_filter_manager")
    @patch("src.ui.handlers.get_api_client")
    def test_fetch_limit_scales_with_max_results(self, mock_client, mock_filter_mgr):
        """Fetch limit should be at least max_results * 5 or 200."""
        client = MagicMock()
        client.fetch_experiments.return_value = pd.DataFrame()
        mock_client.return_value = client
        mock_filter_mgr.return_value = MagicMock()

        # Test with small max_results
        filter_state = FilterState(assay_type="ChIP-seq")
        execute_search(filter_state, max_results=10)

        call_kwargs = client.fetch_experiments.call_args.kwargs
        assert call_kwargs["limit"] >= 200  # max(10*5, 200) = 200

        # Test with larger max_results
        execute_search(filter_state, max_results=100)

        call_kwargs = client.fetch_experiments.call_args.kwargs
        assert call_kwargs["limit"] >= 500  # max(100*5, 200) = 500

    @patch("src.ui.handlers.get_filter_manager")
    @patch("src.ui.handlers.get_api_client")
    def test_all_filter_params_passed_to_client(self, mock_client, mock_filter_mgr):
        """All filter parameters should be passed to the API client."""
        client = MagicMock()
        client.fetch_experiments.return_value = pd.DataFrame()
        mock_client.return_value = client
        mock_filter_mgr.return_value = MagicMock()

        filter_state = FilterState(
            assay_type="ChIP-seq",
            organism="human",
            biosample="liver",
            target="H3K27ac",
            age_stage="adult",
        )
        execute_search(filter_state, max_results=10)

        call_kwargs = client.fetch_experiments.call_args.kwargs
        assert call_kwargs["assay_type"] == "ChIP-seq"
        assert call_kwargs["organism"] == "Homo sapiens"
        assert call_kwargs["biosample"] == "liver"
        assert call_kwargs["target"] == "H3K27ac"
        assert call_kwargs["life_stage"] == "adult"
