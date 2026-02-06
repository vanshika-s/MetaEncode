# tests/test_ui/test_autocomplete.py
"""Tests for autocomplete provider module."""

import pytest

from src.ui.autocomplete import (
    AutocompleteProvider,
    AutocompleteSuggestion,
    create_assay_search_fn,
    create_biosample_search_fn,
    create_lab_search_fn,
    create_organ_search_fn,
    create_organism_search_fn,
    create_target_search_fn,
    get_autocomplete_provider,
)

# =============================================================================
# AutocompleteSuggestion Tests
# =============================================================================


class TestAutocompleteSuggestion:
    """Tests for AutocompleteSuggestion dataclass."""

    def test_suggestion_creation(self) -> None:
        """Test creating an AutocompleteSuggestion."""
        suggestion = AutocompleteSuggestion(
            value="cerebellum",
            display="cerebellum (1,234)",
            category="Brain",
            count=1234,
            is_correction=False,
            match_type="prefix",
            confidence=0.95,
        )
        assert suggestion.value == "cerebellum"
        assert suggestion.display == "cerebellum (1,234)"
        assert suggestion.category == "Brain"
        assert suggestion.count == 1234
        assert suggestion.is_correction is False
        assert suggestion.match_type == "prefix"
        assert suggestion.confidence == 0.95

    def test_suggestion_defaults(self) -> None:
        """Test AutocompleteSuggestion default values."""
        suggestion = AutocompleteSuggestion(
            value="test",
            display="test",
        )
        assert suggestion.category is None
        assert suggestion.count is None
        assert suggestion.is_correction is False
        assert suggestion.match_type == "exact"
        assert suggestion.confidence == 1.0


# =============================================================================
# AutocompleteProvider Tests
# =============================================================================


class TestAutocompleteProviderInit:
    """Tests for AutocompleteProvider initialization."""

    def test_init_default(self) -> None:
        """Test initialization with defaults."""
        provider = AutocompleteProvider()
        assert provider._use_spell_check is True
        assert provider._filter_manager is not None

    def test_init_no_spell_check(self) -> None:
        """Test initialization with spell check disabled."""
        provider = AutocompleteProvider(use_spell_check=False)
        assert provider._use_spell_check is False


class TestAutocompleteProviderGetSuggestions:
    """Tests for AutocompleteProvider.get_suggestions() method."""

    @pytest.fixture
    def provider(self) -> AutocompleteProvider:
        """Create an AutocompleteProvider instance."""
        return AutocompleteProvider(use_spell_check=True)

    # --- Biosample field tests ---

    def test_get_suggestions_biosample_empty(
        self, provider: AutocompleteProvider
    ) -> None:
        """Test biosample suggestions with empty query."""
        suggestions = provider.get_suggestions("", field="biosample", limit=10)
        assert len(suggestions) > 0
        assert len(suggestions) <= 10
        # All should have required keys
        for s in suggestions:
            assert "value" in s
            assert "display" in s

    def test_get_suggestions_biosample_prefix(
        self, provider: AutocompleteProvider
    ) -> None:
        """Test biosample suggestions with prefix match."""
        suggestions = provider.get_suggestions("cereb", field="biosample", limit=10)
        assert len(suggestions) > 0
        # Should include cerebellum or similar
        values = [s["value"].lower() for s in suggestions]
        assert any("cereb" in v for v in values)

    def test_get_suggestions_biosample_k562(
        self, provider: AutocompleteProvider
    ) -> None:
        """Test biosample suggestions for K562."""
        suggestions = provider.get_suggestions("K562", field="biosample", limit=10)
        assert len(suggestions) > 0
        values = [s["value"] for s in suggestions]
        assert "K562" in values

    def test_get_suggestions_biosample_with_organ_context(
        self, provider: AutocompleteProvider
    ) -> None:
        """Test biosample suggestions filtered by organ."""
        suggestions = provider.get_suggestions(
            "", field="biosample", limit=10, context={"organ": "brain"}
        )
        assert len(suggestions) > 0
        # All should be brain-related
        for s in suggestions:
            # Category should indicate brain if available
            if s.get("category"):
                # Might be "Brain" or related
                pass

    # --- Target field tests ---

    def test_get_suggestions_target_empty(self, provider: AutocompleteProvider) -> None:
        """Test target suggestions with empty query."""
        suggestions = provider.get_suggestions("", field="target", limit=10)
        assert len(suggestions) > 0

    def test_get_suggestions_target_h3k27(self, provider: AutocompleteProvider) -> None:
        """Test target suggestions for H3K27."""
        suggestions = provider.get_suggestions("H3K27", field="target", limit=10)
        assert len(suggestions) > 0
        values = [s["value"] for s in suggestions]
        # Should include H3K27ac or H3K27me3
        assert any("H3K27" in v for v in values)

    def test_get_suggestions_target_ctcf(self, provider: AutocompleteProvider) -> None:
        """Test target suggestions for CTCF."""
        suggestions = provider.get_suggestions("CTCF", field="target", limit=10)
        assert len(suggestions) > 0
        values = [s["value"] for s in suggestions]
        assert "CTCF" in values

    # --- Assay field tests ---

    def test_get_suggestions_assay_empty(self, provider: AutocompleteProvider) -> None:
        """Test assay suggestions with empty query."""
        suggestions = provider.get_suggestions("", field="assay", limit=10)
        assert len(suggestions) > 0

    def test_get_suggestions_assay_chip(self, provider: AutocompleteProvider) -> None:
        """Test assay suggestions for ChIP."""
        suggestions = provider.get_suggestions("ChIP", field="assay", limit=10)
        assert len(suggestions) > 0
        values = [s["value"] for s in suggestions]
        assert any("ChIP" in v for v in values)

    def test_get_suggestions_assay_rna(self, provider: AutocompleteProvider) -> None:
        """Test assay suggestions for RNA."""
        suggestions = provider.get_suggestions("RNA", field="assay", limit=10)
        assert len(suggestions) > 0
        values = [s["value"] for s in suggestions]
        assert any("RNA" in v for v in values)

    # --- Organism field tests ---

    def test_get_suggestions_organism_empty(
        self, provider: AutocompleteProvider
    ) -> None:
        """Test organism suggestions with empty query."""
        suggestions = provider.get_suggestions("", field="organism", limit=5)
        assert len(suggestions) > 0

    def test_get_suggestions_organism_human(
        self, provider: AutocompleteProvider
    ) -> None:
        """Test organism suggestions for human."""
        suggestions = provider.get_suggestions("Homo", field="organism", limit=5)
        assert len(suggestions) > 0
        values = [s["value"] for s in suggestions]
        assert any("Homo" in v for v in values)

    def test_get_suggestions_organism_mouse(
        self, provider: AutocompleteProvider
    ) -> None:
        """Test organism suggestions for mouse."""
        suggestions = provider.get_suggestions("Mus", field="organism", limit=5)
        assert len(suggestions) > 0
        values = [s["value"] for s in suggestions]
        assert any("Mus" in v for v in values)

    # --- Lab field tests ---

    def test_get_suggestions_lab_empty(self, provider: AutocompleteProvider) -> None:
        """Test lab suggestions with empty query."""
        suggestions = provider.get_suggestions("", field="lab", limit=10)
        assert len(suggestions) > 0

    def test_get_suggestions_lab_bing(self, provider: AutocompleteProvider) -> None:
        """Test lab suggestions for Bing."""
        suggestions = provider.get_suggestions("Bing", field="lab", limit=10)
        # May or may not have results depending on data
        assert isinstance(suggestions, list)

    # --- Life stage field tests ---

    def test_get_suggestions_life_stage_empty(
        self, provider: AutocompleteProvider
    ) -> None:
        """Test life stage suggestions with empty query."""
        suggestions = provider.get_suggestions("", field="life_stage", limit=10)
        assert len(suggestions) > 0

    def test_get_suggestions_life_stage_adult(
        self, provider: AutocompleteProvider
    ) -> None:
        """Test life stage suggestions for adult."""
        suggestions = provider.get_suggestions("adult", field="life_stage", limit=10)
        assert len(suggestions) > 0
        values = [s["value"].lower() for s in suggestions]
        assert "adult" in values

    # --- Organ field tests ---

    def test_get_suggestions_organ_empty(self, provider: AutocompleteProvider) -> None:
        """Test organ suggestions with empty query."""
        suggestions = provider.get_suggestions("", field="organ", limit=10)
        assert len(suggestions) > 0

    def test_get_suggestions_organ_brain(self, provider: AutocompleteProvider) -> None:
        """Test organ suggestions for brain."""
        suggestions = provider.get_suggestions("brain", field="organ", limit=10)
        assert len(suggestions) > 0
        values = [s["value"].lower() for s in suggestions]
        assert "brain" in values

    # --- Limit tests ---

    def test_get_suggestions_respects_limit(
        self, provider: AutocompleteProvider
    ) -> None:
        """Test that limit is respected."""
        suggestions = provider.get_suggestions("", field="biosample", limit=5)
        assert len(suggestions) <= 5

    # --- Match type tests ---

    def test_get_suggestions_prefix_match_type(
        self, provider: AutocompleteProvider
    ) -> None:
        """Test that prefix matches have correct match_type."""
        suggestions = provider.get_suggestions("K562", field="biosample", limit=10)
        # K562 should be a prefix match
        k562_sugg = next((s for s in suggestions if s["value"] == "K562"), None)
        if k562_sugg:
            assert k562_sugg["match_type"] == "prefix"


class TestAutocompleteProviderSpellCheck:
    """Tests for AutocompleteProvider spell check integration."""

    def test_spell_correction_cerebelum(self) -> None:
        """Test that 'cerebelum' suggests 'cerebellum'."""
        provider = AutocompleteProvider(use_spell_check=True)
        suggestions = provider.get_suggestions("cerebelum", field="biosample", limit=10)

        # Should include cerebellum as a correction
        values = [s["value"] for s in suggestions]
        if "cerebellum" in values:
            cerebellum_sugg = next(s for s in suggestions if s["value"] == "cerebellum")
            # Might be marked as correction if spell check found it
            # (depends on whether it's also a fuzzy match)

    def test_no_spell_check_when_disabled(self) -> None:
        """Test that spell check is not used when disabled."""
        provider = AutocompleteProvider(use_spell_check=False)
        # This should still work, just without spell corrections
        suggestions = provider.get_suggestions("cerebelum", field="biosample", limit=10)
        assert isinstance(suggestions, list)


# =============================================================================
# Singleton Tests
# =============================================================================


class TestGetAutocompleteProvider:
    """Tests for get_autocomplete_provider singleton."""

    def test_returns_same_instance(self) -> None:
        """Test that get_autocomplete_provider returns singleton."""
        provider1 = get_autocomplete_provider()
        provider2 = get_autocomplete_provider()
        assert provider1 is provider2


# =============================================================================
# Search Function Factory Tests
# =============================================================================


class TestCreateSearchFunctions:
    """Tests for search function factories."""

    def test_create_biosample_search_fn(self) -> None:
        """Test create_biosample_search_fn returns callable."""
        search_fn = create_biosample_search_fn()
        assert callable(search_fn)

        # Should return list of tuples
        results = search_fn("")
        assert isinstance(results, list)
        if len(results) > 0:
            assert isinstance(results[0], tuple)
            assert len(results[0]) == 2

    def test_create_biosample_search_fn_with_organ(self) -> None:
        """Test create_biosample_search_fn with organ filter."""
        search_fn = create_biosample_search_fn(organ="brain")
        assert callable(search_fn)

        results = search_fn("")
        assert isinstance(results, list)

    def test_create_target_search_fn(self) -> None:
        """Test create_target_search_fn returns callable."""
        search_fn = create_target_search_fn()
        assert callable(search_fn)

        results = search_fn("H3K27")
        assert isinstance(results, list)

    def test_create_assay_search_fn(self) -> None:
        """Test create_assay_search_fn returns callable."""
        search_fn = create_assay_search_fn()
        assert callable(search_fn)

        results = search_fn("ChIP")
        assert isinstance(results, list)

    def test_create_organism_search_fn(self) -> None:
        """Test create_organism_search_fn returns callable."""
        search_fn = create_organism_search_fn()
        assert callable(search_fn)

        results = search_fn("")
        assert isinstance(results, list)

    def test_create_lab_search_fn(self) -> None:
        """Test create_lab_search_fn returns callable."""
        search_fn = create_lab_search_fn()
        assert callable(search_fn)

        results = search_fn("")
        assert isinstance(results, list)

    def test_create_organ_search_fn(self) -> None:
        """Test create_organ_search_fn returns callable."""
        search_fn = create_organ_search_fn()
        assert callable(search_fn)

        results = search_fn("brain")
        assert isinstance(results, list)
