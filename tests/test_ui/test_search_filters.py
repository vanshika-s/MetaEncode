# tests/test_ui/test_search_filters.py
"""Tests for search filter utilities."""

import pandas as pd
import pytest

from src.ui.search_filters import (
    FilterState,
    SearchFilterManager,
    parse_age_from_text,
)

# =============================================================================
# FilterState Tests
# =============================================================================


class TestFilterStateInit:
    """Tests for FilterState initialization."""

    def test_default_initialization(self) -> None:
        """Test that FilterState initializes with correct defaults."""
        state = FilterState()
        assert state.assay_type is None
        assert state.organism is None
        assert state.body_part is None
        assert state.biosample is None
        assert state.target is None
        assert state.age_stage is None
        assert state.lab is None
        assert state.min_replicates == 0
        assert state.max_results == 20
        assert state.description_search is None

    def test_custom_initialization(self) -> None:
        """Test FilterState with custom values."""
        state = FilterState(
            assay_type="ChIP-seq",
            organism="mouse",
            biosample="cerebellum",
            target="H3K27ac",
            max_results=25,
        )
        assert state.assay_type == "ChIP-seq"
        assert state.organism == "mouse"
        assert state.biosample == "cerebellum"
        assert state.target == "H3K27ac"
        assert state.max_results == 25


class TestFilterStateToDict:
    """Tests for FilterState.to_dict() method."""

    def test_to_dict_default(self) -> None:
        """Test to_dict with default values."""
        state = FilterState()
        result = state.to_dict()

        assert isinstance(result, dict)
        assert result["assay_type"] is None
        assert result["organism"] is None
        assert result["min_replicates"] == 0
        assert result["max_results"] == 20

    def test_to_dict_with_values(self) -> None:
        """Test to_dict with custom values."""
        state = FilterState(
            assay_type="RNA-seq",
            organism="human",
            target="H3K4me3",
            min_replicates=2,
        )
        result = state.to_dict()

        assert result["assay_type"] == "RNA-seq"
        assert result["organism"] == "human"
        assert result["target"] == "H3K4me3"
        assert result["min_replicates"] == 2

    def test_to_dict_contains_all_fields(self) -> None:
        """Test that to_dict contains all expected fields."""
        state = FilterState()
        result = state.to_dict()

        expected_keys = {
            "assay_type",
            "organism",
            "body_part",
            "biosample",
            "target",
            "age_stage",
            "lab",
            "min_replicates",
            "max_results",
            "description_search",
        }
        assert set(result.keys()) == expected_keys


class TestFilterStateFromDict:
    """Tests for FilterState.from_dict() class method."""

    def test_from_dict_empty(self) -> None:
        """Test from_dict with empty dictionary."""
        state = FilterState.from_dict({})
        assert state.assay_type is None
        assert state.min_replicates == 0
        assert state.max_results == 20

    def test_from_dict_partial(self) -> None:
        """Test from_dict with partial dictionary."""
        data = {"assay_type": "ATAC-seq", "organism": "mouse"}
        state = FilterState.from_dict(data)

        assert state.assay_type == "ATAC-seq"
        assert state.organism == "mouse"
        assert state.target is None

    def test_from_dict_roundtrip(self) -> None:
        """Test that to_dict and from_dict are inverses."""
        original = FilterState(
            assay_type="ChIP-seq",
            organism="human",
            biosample="K562",
            target="CTCF",
            age_stage="adult",
            lab="Bing Ren",
            min_replicates=2,
            max_results=30,
            description_search="enhancer",
        )
        data = original.to_dict()
        restored = FilterState.from_dict(data)

        assert restored.assay_type == original.assay_type
        assert restored.organism == original.organism
        assert restored.biosample == original.biosample
        assert restored.target == original.target
        assert restored.age_stage == original.age_stage
        assert restored.lab == original.lab
        assert restored.min_replicates == original.min_replicates
        assert restored.max_results == original.max_results
        assert restored.description_search == original.description_search


class TestFilterStateHasAnyFilter:
    """Tests for FilterState.has_any_filter() method."""

    def test_has_any_filter_empty(self) -> None:
        """Test has_any_filter with no filters set."""
        state = FilterState()
        assert state.has_any_filter() is False

    def test_has_any_filter_with_assay_type(self) -> None:
        """Test has_any_filter with assay_type set."""
        state = FilterState(assay_type="ChIP-seq")
        assert state.has_any_filter() is True

    def test_has_any_filter_with_organism(self) -> None:
        """Test has_any_filter with organism set."""
        state = FilterState(organism="mouse")
        assert state.has_any_filter() is True

    def test_has_any_filter_with_body_part(self) -> None:
        """Test has_any_filter with body_part set."""
        state = FilterState(body_part="brain")
        assert state.has_any_filter() is True

    def test_has_any_filter_with_biosample(self) -> None:
        """Test has_any_filter with biosample set."""
        state = FilterState(biosample="cerebellum")
        assert state.has_any_filter() is True

    def test_has_any_filter_with_target(self) -> None:
        """Test has_any_filter with target set."""
        state = FilterState(target="H3K27ac")
        assert state.has_any_filter() is True

    def test_has_any_filter_with_age_stage(self) -> None:
        """Test has_any_filter with age_stage set."""
        state = FilterState(age_stage="P60")
        assert state.has_any_filter() is True

    def test_has_any_filter_with_lab(self) -> None:
        """Test has_any_filter with lab set."""
        state = FilterState(lab="Bing Ren")
        assert state.has_any_filter() is True

    def test_has_any_filter_with_min_replicates(self) -> None:
        """Test has_any_filter with min_replicates > 0."""
        state = FilterState(min_replicates=2)
        assert state.has_any_filter() is True

    def test_has_any_filter_with_min_replicates_zero(self) -> None:
        """Test has_any_filter with min_replicates = 0."""
        state = FilterState(min_replicates=0)
        assert state.has_any_filter() is False

    def test_has_any_filter_with_description_search(self) -> None:
        """Test has_any_filter with description_search set."""
        state = FilterState(description_search="8-week cerebellum")
        assert state.has_any_filter() is True

    def test_has_any_filter_max_results_only(self) -> None:
        """Test that max_results alone doesn't count as a filter."""
        state = FilterState(max_results=50)
        assert state.has_any_filter() is False


# =============================================================================
# SearchFilterManager Tests
# =============================================================================


class TestSearchFilterManagerInit:
    """Tests for SearchFilterManager initialization."""

    def test_init_creates_instance(self) -> None:
        """Test that SearchFilterManager initializes correctly."""
        manager = SearchFilterManager()
        assert manager is not None

    def test_init_builds_tissue_index(self) -> None:
        """Test that initialization builds tissue-to-body-part index."""
        manager = SearchFilterManager()
        assert hasattr(manager, "_tissue_to_body_part")
        assert isinstance(manager._tissue_to_body_part, dict)
        assert len(manager._tissue_to_body_part) > 0

    def test_init_builds_all_tissues_list(self) -> None:
        """Test that initialization builds all tissues list."""
        manager = SearchFilterManager()
        assert hasattr(manager, "_all_tissues")
        assert isinstance(manager._all_tissues, list)
        assert len(manager._all_tissues) > 0

    def test_init_builds_synonym_map(self) -> None:
        """Test that initialization builds synonym map."""
        manager = SearchFilterManager()
        assert hasattr(manager, "_synonym_map")
        assert isinstance(manager._synonym_map, dict)
        # Check cerebellum/hindbrain are in synonym map
        assert "cerebellum" in manager._synonym_map
        assert "hindbrain" in manager._synonym_map


class TestSearchFilterManagerAutocompleteAssay:
    """Tests for SearchFilterManager.autocomplete_assay() method."""

    @pytest.fixture
    def manager(self) -> SearchFilterManager:
        """Create SearchFilterManager instance."""
        return SearchFilterManager()

    def test_autocomplete_assay_empty_query(self, manager: SearchFilterManager) -> None:
        """Test autocomplete_assay with empty query returns common assays."""
        results = manager.autocomplete_assay("")
        assert len(results) > 0
        # Should return common assay types
        keys = [r[0] for r in results]
        assert "ChIP-seq" in keys or "RNA-seq" in keys

    def test_autocomplete_assay_chip_query(self, manager: SearchFilterManager) -> None:
        """Test autocomplete_assay with 'chip' query."""
        results = manager.autocomplete_assay("chip")
        assert len(results) > 0
        # ChIP-seq should be in results
        keys = [r[0] for r in results]
        assert "ChIP-seq" in keys

    def test_autocomplete_assay_rna_query(self, manager: SearchFilterManager) -> None:
        """Test autocomplete_assay with 'rna' query."""
        results = manager.autocomplete_assay("rna")
        assert len(results) > 0
        keys = [r[0] for r in results]
        assert "RNA-seq" in keys

    def test_autocomplete_assay_hic_query(self, manager: SearchFilterManager) -> None:
        """Test autocomplete_assay with 'hic' query."""
        results = manager.autocomplete_assay("hic")
        assert len(results) > 0
        keys = [r[0] for r in results]
        # Should match Hi-C or HiC
        assert any("Hi-C" in k or "HiC" in k for k in keys)

    def test_autocomplete_assay_limit(self, manager: SearchFilterManager) -> None:
        """Test that autocomplete_assay respects limit parameter."""
        results = manager.autocomplete_assay("seq", limit=3)
        assert len(results) <= 3

    def test_autocomplete_assay_returns_tuples(
        self, manager: SearchFilterManager
    ) -> None:
        """Test that autocomplete_assay returns list of tuples."""
        results = manager.autocomplete_assay("atac")
        assert len(results) > 0
        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], str)  # key
            assert isinstance(result[1], str)  # display

    def test_autocomplete_assay_no_match(self, manager: SearchFilterManager) -> None:
        """Test autocomplete_assay with no matching query."""
        results = manager.autocomplete_assay("xyz123nonexistent")
        # Should return empty or very few results
        assert len(results) < 3


class TestSearchFilterManagerAutocompleteOrganism:
    """Tests for SearchFilterManager.autocomplete_organism() method."""

    @pytest.fixture
    def manager(self) -> SearchFilterManager:
        """Create SearchFilterManager instance."""
        return SearchFilterManager()

    def test_autocomplete_organism_empty_query(
        self, manager: SearchFilterManager
    ) -> None:
        """Test autocomplete_organism with empty query returns organisms."""
        results = manager.autocomplete_organism("")
        assert len(results) > 0
        keys = [r[0] for r in results]
        # Now returns scientific names
        assert "Homo sapiens" in keys
        assert "Mus musculus" in keys

    def test_autocomplete_organism_human_query(
        self, manager: SearchFilterManager
    ) -> None:
        """Test autocomplete_organism with 'human' query."""
        results = manager.autocomplete_organism("human")
        assert len(results) > 0
        keys = [r[0] for r in results]
        # Returns scientific name when searching for common name
        assert "Homo sapiens" in keys

    def test_autocomplete_organism_mouse_query(
        self, manager: SearchFilterManager
    ) -> None:
        """Test autocomplete_organism with 'mouse' query."""
        results = manager.autocomplete_organism("mouse")
        assert len(results) > 0
        keys = [r[0] for r in results]
        # Returns scientific name when searching for common name
        assert "Mus musculus" in keys

    def test_autocomplete_organism_assembly_query(
        self, manager: SearchFilterManager
    ) -> None:
        """Test autocomplete_organism with genome assembly query."""
        results = manager.autocomplete_organism("hg38")
        assert len(results) > 0
        keys = [r[0] for r in results]
        # Returns scientific name when searching by assembly
        assert "Homo sapiens" in keys

    def test_autocomplete_organism_includes_assembly_in_display(
        self, manager: SearchFilterManager
    ) -> None:
        """Test that organism display includes genome assembly."""
        results = manager.autocomplete_organism("human")
        assert len(results) > 0
        # Find the Homo sapiens result
        for key, display in results:
            if key == "Homo sapiens":
                assert "hg38" in display


class TestSearchFilterManagerAutocompleteTarget:
    """Tests for SearchFilterManager.autocomplete_target() method."""

    @pytest.fixture
    def manager(self) -> SearchFilterManager:
        """Create SearchFilterManager instance."""
        return SearchFilterManager()

    def test_autocomplete_target_empty_query(
        self, manager: SearchFilterManager
    ) -> None:
        """Test autocomplete_target with empty query returns common targets."""
        results = manager.autocomplete_target("")
        assert len(results) > 0
        keys = [r[0] for r in results]
        assert "H3K27ac" in keys

    def test_autocomplete_target_h3k27_query(
        self, manager: SearchFilterManager
    ) -> None:
        """Test autocomplete_target with 'H3K27' query."""
        results = manager.autocomplete_target("H3K27")
        assert len(results) > 0
        keys = [r[0] for r in results]
        # Should match H3K27ac and H3K27me3
        assert "H3K27ac" in keys or "H3K27me3" in keys

    def test_autocomplete_target_ctcf_query(self, manager: SearchFilterManager) -> None:
        """Test autocomplete_target with 'ctcf' query."""
        results = manager.autocomplete_target("ctcf")
        assert len(results) > 0
        keys = [r[0] for r in results]
        assert "CTCF" in keys

    def test_autocomplete_target_alias_query(
        self, manager: SearchFilterManager
    ) -> None:
        """Test autocomplete_target with alias like 'polycomb'."""
        results = manager.autocomplete_target("polycomb")
        assert len(results) > 0
        keys = [r[0] for r in results]
        # Should match H3K27me3 (Polycomb repression)
        assert "H3K27me3" in keys


class TestSearchFilterManagerAutocompleteBodyPart:
    """Tests for SearchFilterManager.autocomplete_body_part() method."""

    @pytest.fixture
    def manager(self) -> SearchFilterManager:
        """Create SearchFilterManager instance."""
        return SearchFilterManager()

    def test_autocomplete_body_part_empty_query(
        self, manager: SearchFilterManager
    ) -> None:
        """Test autocomplete_body_part with empty query returns all body parts."""
        results = manager.autocomplete_body_part("")
        assert len(results) > 0
        keys = [r[0] for r in results]
        assert "brain" in keys

    def test_autocomplete_body_part_brain_query(
        self, manager: SearchFilterManager
    ) -> None:
        """Test autocomplete_body_part with 'brain' query."""
        results = manager.autocomplete_body_part("brain")
        assert len(results) > 0
        keys = [r[0] for r in results]
        assert "brain" in keys

    def test_autocomplete_body_part_alias_query(
        self, manager: SearchFilterManager
    ) -> None:
        """Test autocomplete_body_part with alias like 'cardiac'."""
        results = manager.autocomplete_body_part("cardiac")
        assert len(results) > 0
        keys = [r[0] for r in results]
        assert "heart" in keys

    def test_autocomplete_body_part_tissue_match(
        self, manager: SearchFilterManager
    ) -> None:
        """Test autocomplete_body_part matching a tissue name."""
        results = manager.autocomplete_body_part("cerebellum")
        assert len(results) > 0
        keys = [r[0] for r in results]
        # Should match brain since cerebellum is a brain tissue
        assert "brain" in keys


class TestSearchFilterManagerAutocompleteBiosample:
    """Tests for SearchFilterManager.autocomplete_biosample() method."""

    @pytest.fixture
    def manager(self) -> SearchFilterManager:
        """Create SearchFilterManager instance."""
        return SearchFilterManager()

    def test_autocomplete_biosample_empty_query(
        self, manager: SearchFilterManager
    ) -> None:
        """Test autocomplete_biosample with empty query."""
        results = manager.autocomplete_biosample("")
        assert len(results) > 0

    def test_autocomplete_biosample_skips_duplicates(
        self, manager: SearchFilterManager
    ) -> None:
        """Test that autocomplete_biosample skips duplicate tissues (line 350)."""
        # Search for a tissue that appears in the list
        # The function should handle duplicates gracefully
        results = manager.autocomplete_biosample("brain")
        # Get the tissue names from results
        tissues = [r[0] for r in results]
        # No duplicates should exist
        assert len(tissues) == len(set(t.lower() for t in tissues))

    def test_autocomplete_biosample_cerebellum_query(
        self, manager: SearchFilterManager
    ) -> None:
        """Test autocomplete_biosample with 'cerebellum' query."""
        results = manager.autocomplete_biosample("cerebellum")
        assert len(results) > 0
        tissues = [r[0] for r in results]
        assert "cerebellum" in tissues

    def test_autocomplete_biosample_with_body_part(
        self, manager: SearchFilterManager
    ) -> None:
        """Test autocomplete_biosample restricted to body part."""
        results = manager.autocomplete_biosample("", body_part="brain")
        assert len(results) > 0
        # All results should be brain tissues
        for tissue, body_part_display in results:
            assert "Brain" in body_part_display or "Nervous" in body_part_display

    def test_autocomplete_biosample_synonym_match(
        self, manager: SearchFilterManager
    ) -> None:
        """Test autocomplete_biosample matches synonyms."""
        results = manager.autocomplete_biosample("hindbrain")
        assert len(results) > 0
        # Should match hindbrain or related tissues like cerebellum
        tissues = [r[0] for r in results]
        assert "hindbrain" in tissues or "cerebellum" in tissues

    def test_autocomplete_biosample_cell_line(
        self, manager: SearchFilterManager
    ) -> None:
        """Test autocomplete_biosample with cell line query."""
        results = manager.autocomplete_biosample("K562")
        assert len(results) > 0
        tissues = [r[0] for r in results]
        assert "K562" in tissues

    def test_autocomplete_biosample_invalid_body_part(
        self, manager: SearchFilterManager
    ) -> None:
        """Test autocomplete_biosample with invalid body_part defaults to all."""
        results = manager.autocomplete_biosample("", body_part="invalid_part")
        assert len(results) > 0


class TestSearchFilterManagerAutocompleteAge:
    """Tests for SearchFilterManager.autocomplete_age() method.

    Note: These tests use ACTUAL ENCODE life stages (adult, embryonic, child, etc.),
    not fabricated developmental stages like P60 or E14.5 which don't exist in ENCODE.
    """

    @pytest.fixture
    def manager(self) -> SearchFilterManager:
        """Create SearchFilterManager instance."""
        return SearchFilterManager()

    def test_autocomplete_age_empty_query(self, manager: SearchFilterManager) -> None:
        """Test autocomplete_age with empty query returns real ENCODE life stages."""
        results = manager.autocomplete_age("")
        assert len(results) > 0
        # Results should be (stage_name, count_info) tuples
        stages = [r[0] for r in results]
        # Should contain real ENCODE life stages
        assert "adult" in stages
        assert "embryonic" in stages

    def test_autocomplete_age_adult_query(self, manager: SearchFilterManager) -> None:
        """Test autocomplete_age with 'adult' query."""
        results = manager.autocomplete_age("adult")
        assert len(results) > 0
        stages = [r[0] for r in results]
        assert "adult" in stages

    def test_autocomplete_age_embryonic_query(
        self, manager: SearchFilterManager
    ) -> None:
        """Test autocomplete_age with 'embryonic' query."""
        results = manager.autocomplete_age("embryonic")
        assert len(results) > 0
        stages = [r[0] for r in results]
        assert "embryonic" in stages

    def test_autocomplete_age_child_query(self, manager: SearchFilterManager) -> None:
        """Test autocomplete_age with 'child' query."""
        results = manager.autocomplete_age("child")
        assert len(results) > 0
        stages = [r[0] for r in results]
        assert "child" in stages

    def test_autocomplete_age_organism_param_ignored(
        self, manager: SearchFilterManager
    ) -> None:
        """Test that organism parameter doesn't filter (life stages are cross-organism).

        Note: ENCODE's life_stage field is not organism-specific. The same stages
        (adult, embryonic, etc.) apply across species. Unlike fabricated stages
        like P60 (mouse-specific), real life stages are universal.
        """
        results_human = manager.autocomplete_age("", organism="human")
        results_mouse = manager.autocomplete_age("", organism="mouse")
        results_none = manager.autocomplete_age("")

        # All should return the same results (organism doesn't filter life stages)
        assert len(results_human) == len(results_none)
        assert len(results_mouse) == len(results_none)

    def test_autocomplete_age_newborn_query(self, manager: SearchFilterManager) -> None:
        """Test autocomplete_age with 'newborn' query."""
        results = manager.autocomplete_age("newborn")
        assert len(results) > 0
        stages = [r[0] for r in results]
        assert "newborn" in stages

    def test_autocomplete_age_returns_count_info(
        self, manager: SearchFilterManager
    ) -> None:
        """Test that autocomplete_age returns experiment counts in description."""
        results = manager.autocomplete_age("")
        assert len(results) > 0
        # Second element should be count info like "25,196 experiments"
        stage, count_info = results[0]
        assert "experiments" in count_info


class TestSearchFilterManagerAutocompleteLab:
    """Tests for SearchFilterManager.autocomplete_lab() method."""

    @pytest.fixture
    def manager(self) -> SearchFilterManager:
        """Create SearchFilterManager instance."""
        return SearchFilterManager()

    def test_autocomplete_lab_empty_query(self, manager: SearchFilterManager) -> None:
        """Test autocomplete_lab with empty query returns labs."""
        results = manager.autocomplete_lab("")
        assert len(results) > 0
        assert isinstance(results[0], str)

    def test_autocomplete_lab_bing_query(self, manager: SearchFilterManager) -> None:
        """Test autocomplete_lab with 'Bing' query."""
        results = manager.autocomplete_lab("Bing")
        assert len(results) > 0
        assert any("Bing" in lab for lab in results)

    def test_autocomplete_lab_stanford_query(
        self, manager: SearchFilterManager
    ) -> None:
        """Test autocomplete_lab with 'Stanford' query."""
        results = manager.autocomplete_lab("Stanford")
        assert len(results) > 0
        assert any("Stanford" in lab for lab in results)


class TestSearchFilterManagerGetRelatedTissues:
    """Tests for SearchFilterManager.get_related_tissues() method."""

    @pytest.fixture
    def manager(self) -> SearchFilterManager:
        """Create SearchFilterManager instance."""
        return SearchFilterManager()

    def test_get_related_tissues_cerebellum(self, manager: SearchFilterManager) -> None:
        """Test get_related_tissues for cerebellum."""
        related = manager.get_related_tissues("cerebellum")
        assert isinstance(related, list)
        assert "cerebellum" in related
        assert "hindbrain" in related

    def test_get_related_tissues_hindbrain(self, manager: SearchFilterManager) -> None:
        """Test get_related_tissues for hindbrain."""
        related = manager.get_related_tissues("hindbrain")
        assert isinstance(related, list)
        assert "hindbrain" in related
        assert "cerebellum" in related

    def test_get_related_tissues_no_synonyms(
        self, manager: SearchFilterManager
    ) -> None:
        """Test get_related_tissues for tissue without synonyms."""
        related = manager.get_related_tissues("K562")
        assert isinstance(related, list)
        assert "K562" in related

    def test_get_related_tissues_unknown(self, manager: SearchFilterManager) -> None:
        """Test get_related_tissues for unknown tissue."""
        related = manager.get_related_tissues("unknown_tissue")
        assert isinstance(related, list)
        assert "unknown_tissue" in related
        assert len(related) == 1


class TestSearchFilterManagerMatchScore:
    """Tests for SearchFilterManager._match_score() method."""

    @pytest.fixture
    def manager(self) -> SearchFilterManager:
        """Create SearchFilterManager instance."""
        return SearchFilterManager()

    def test_match_score_exact(self, manager: SearchFilterManager) -> None:
        """Test _match_score for exact match."""
        score = manager._match_score("chipseq", "chipseq")
        assert score == 1.0

    def test_match_score_prefix(self, manager: SearchFilterManager) -> None:
        """Test _match_score for prefix match."""
        score = manager._match_score("chip", "chipseq")
        assert score > 0.8

    def test_match_score_contains(self, manager: SearchFilterManager) -> None:
        """Test _match_score for substring match."""
        score = manager._match_score("seq", "chipseq")
        assert score > 0.5

    def test_match_score_empty_query(self, manager: SearchFilterManager) -> None:
        """Test _match_score with empty query."""
        score = manager._match_score("", "chipseq")
        assert score == 0.0

    def test_match_score_empty_target(self, manager: SearchFilterManager) -> None:
        """Test _match_score with empty target."""
        score = manager._match_score("chip", "")
        assert score == 0.0

    def test_match_score_no_match(self, manager: SearchFilterManager) -> None:
        """Test _match_score for no match."""
        score = manager._match_score("xyz", "abc")
        assert score < 0.5

    def test_match_score_query_contains_target(
        self, manager: SearchFilterManager
    ) -> None:
        """Test _match_score where query contains target."""
        score = manager._match_score("chipseq analysis", "chip")
        assert score > 0.5

    def test_match_score_word_boundary(self, manager: SearchFilterManager) -> None:
        """Test _match_score for word boundary match."""
        score = manager._match_score("chip", "chip seq analysis")
        assert score > 0.6

    def test_match_score_query_starts_with_target(
        self, manager: SearchFilterManager
    ) -> None:
        """Test _match_score where query starts with shorter target."""
        score = manager._match_score("chipseq", "chip")
        assert score > 0.7

    def test_match_score_word_boundary_partial_match(
        self, manager: SearchFilterManager
    ) -> None:
        """Test _match_score with word boundary match (line 505)."""
        # Query "rna" matches word "rna" in "rna seq analysis"
        score = manager._match_score("rna", "rna seq analysis")
        assert score >= 0.65

    def test_match_score_fuzzy_only(self, manager: SearchFilterManager) -> None:
        """Test _match_score falls back to fuzzy matching for dissimilar strings."""
        # These strings don't match by prefix, substring, or word boundary
        score = manager._match_score("abcd", "wxyz")
        assert score < 0.5  # Should be low fuzzy score

    def test_match_score_substring_match_internal(
        self, manager: SearchFilterManager
    ) -> None:
        """Test _match_score with substring that matches an internal word.

        Note: Line 505 (word boundary match returning 0.65) is logically
        unreachable because if word.startswith(query), then query is
        necessarily a substring of target, which is caught by line 495 first.
        """
        # "seq" is a substring of "chip seq" (caught by line 495)
        score = manager._match_score("seq", "chip seq")
        # Should hit line 495 (substring match), returning 0.7 + 0.1 * (3/8) ≈ 0.7375
        assert 0.7 < score < 0.8


class TestSearchFilterManagerGetBodyPartDisplay:
    """Tests for SearchFilterManager._get_body_part_display() method."""

    @pytest.fixture
    def manager(self) -> SearchFilterManager:
        """Create SearchFilterManager instance."""
        return SearchFilterManager()

    def test_get_body_part_display_cerebellum(
        self, manager: SearchFilterManager
    ) -> None:
        """Test _get_body_part_display for cerebellum."""
        display = manager._get_body_part_display("cerebellum")
        assert "Brain" in display or "Nervous" in display

    def test_get_body_part_display_k562(self, manager: SearchFilterManager) -> None:
        """Test _get_body_part_display for K562."""
        display = manager._get_body_part_display("K562")
        assert "Cell Line" in display

    def test_get_body_part_display_unknown(self, manager: SearchFilterManager) -> None:
        """Test _get_body_part_display for unknown tissue."""
        display = manager._get_body_part_display("unknown_tissue")
        assert display == ""


# =============================================================================
# Apply Filters Tests
# =============================================================================


@pytest.fixture
def sample_df_for_filtering() -> pd.DataFrame:
    """Create sample DataFrame for filter testing."""
    return pd.DataFrame(
        {
            "accession": ["ENC001", "ENC002", "ENC003", "ENC004", "ENC005"],
            "assay_term_name": ["ChIP-seq", "RNA-seq", "Hi-C", "ChIP-seq", "ATAC-seq"],
            "organism": ["human", "mouse", "human", "mouse", "human"],
            "biosample_term_name": [
                "K562",
                "liver",
                "cerebellum",
                "hindbrain",
                "HepG2",
            ],
            "description": [
                "ChIP-seq targeting H3K27ac in K562",
                "RNA-seq of mouse liver tissue",
                "Hi-C on P60 mouse cerebellum",
                "ChIP-seq targeting H3K4me3 in 8-week hindbrain",
                "ATAC-seq on adult HepG2 cells",
            ],
            "life_stage": ["adult", "embryonic", "postnatal", "adult", "adult"],
            "lab": ["lab-a", "lab-b", "Bing Ren", "lab-b", "lab-a"],
            "replicate_count": [2, 3, 1, 4, 2],
        }
    )


class TestSearchFilterManagerApplyFilters:
    """Tests for SearchFilterManager.apply_filters() method."""

    @pytest.fixture
    def manager(self) -> SearchFilterManager:
        """Create SearchFilterManager instance."""
        return SearchFilterManager()

    def test_apply_filters_empty_df(self, manager: SearchFilterManager) -> None:
        """Test apply_filters with empty DataFrame."""
        df = pd.DataFrame()
        filters = FilterState(organism="human")
        result = manager.apply_filters(df, filters)
        assert result.empty

    def test_apply_filters_no_filters(
        self, manager: SearchFilterManager, sample_df_for_filtering: pd.DataFrame
    ) -> None:
        """Test apply_filters with no filters set."""
        filters = FilterState()
        result = manager.apply_filters(sample_df_for_filtering, filters)
        assert len(result) == len(sample_df_for_filtering)

    def test_apply_filters_organism(
        self, manager: SearchFilterManager, sample_df_for_filtering: pd.DataFrame
    ) -> None:
        """Test apply_filters with organism filter."""
        filters = FilterState(organism="human")
        result = manager.apply_filters(sample_df_for_filtering, filters)
        assert len(result) == 3
        assert all(result["organism"].str.lower() == "human")

    def test_apply_filters_assay_type(
        self, manager: SearchFilterManager, sample_df_for_filtering: pd.DataFrame
    ) -> None:
        """Test apply_filters with assay_type filter."""
        filters = FilterState(assay_type="ChIP-seq")
        result = manager.apply_filters(sample_df_for_filtering, filters)
        assert len(result) == 2
        assert all(result["assay_term_name"] == "ChIP-seq")

    def test_apply_filters_hic_variants(
        self, manager: SearchFilterManager, sample_df_for_filtering: pd.DataFrame
    ) -> None:
        """Test apply_filters with Hi-C filter matches variants."""
        filters = FilterState(assay_type="Hi-C")
        result = manager.apply_filters(sample_df_for_filtering, filters)
        assert len(result) == 1
        assert result.iloc[0]["assay_term_name"] == "Hi-C"

    def test_apply_filters_biosample_with_synonym(
        self, manager: SearchFilterManager, sample_df_for_filtering: pd.DataFrame
    ) -> None:
        """Test apply_filters with biosample using synonym expansion."""
        # Filtering for cerebellum should also match hindbrain
        filters = FilterState(biosample="cerebellum")
        result = manager.apply_filters(sample_df_for_filtering, filters)
        assert len(result) == 2
        biosamples = result["biosample_term_name"].tolist()
        assert "cerebellum" in biosamples
        assert "hindbrain" in biosamples

    def test_apply_filters_target(
        self, manager: SearchFilterManager, sample_df_for_filtering: pd.DataFrame
    ) -> None:
        """Test apply_filters with target filter."""
        filters = FilterState(target="H3K27ac")
        result = manager.apply_filters(sample_df_for_filtering, filters)
        assert len(result) == 1
        assert "H3K27ac" in result.iloc[0]["description"]

    def test_apply_filters_age_stage(
        self, manager: SearchFilterManager, sample_df_for_filtering: pd.DataFrame
    ) -> None:
        """Test apply_filters with age_stage filter uses life_stage column."""
        filters = FilterState(age_stage="embryonic")
        result = manager.apply_filters(sample_df_for_filtering, filters)
        assert len(result) == 1
        assert result.iloc[0]["life_stage"] == "embryonic"

    def test_apply_filters_description_search(
        self, manager: SearchFilterManager, sample_df_for_filtering: pd.DataFrame
    ) -> None:
        """Test apply_filters with description search."""
        filters = FilterState(description_search="8-week")
        result = manager.apply_filters(sample_df_for_filtering, filters)
        assert len(result) == 1
        assert "8-week" in result.iloc[0]["description"]

    def test_apply_filters_description_search_multiple_terms(
        self, manager: SearchFilterManager, sample_df_for_filtering: pd.DataFrame
    ) -> None:
        """Test apply_filters with multiple search terms."""
        filters = FilterState(description_search="ChIP-seq K562")
        result = manager.apply_filters(sample_df_for_filtering, filters)
        assert len(result) == 1
        assert "K562" in result.iloc[0]["description"]

    def test_apply_filters_lab(
        self, manager: SearchFilterManager, sample_df_for_filtering: pd.DataFrame
    ) -> None:
        """Test apply_filters with lab filter."""
        filters = FilterState(lab="Bing")
        result = manager.apply_filters(sample_df_for_filtering, filters)
        assert len(result) == 1
        assert "Bing" in result.iloc[0]["lab"]

    def test_apply_filters_min_replicates(
        self, manager: SearchFilterManager, sample_df_for_filtering: pd.DataFrame
    ) -> None:
        """Test apply_filters with min_replicates filter."""
        filters = FilterState(min_replicates=3)
        result = manager.apply_filters(sample_df_for_filtering, filters)
        assert len(result) == 2
        assert all(result["replicate_count"] >= 3)

    def test_apply_filters_combined(
        self, manager: SearchFilterManager, sample_df_for_filtering: pd.DataFrame
    ) -> None:
        """Test apply_filters with multiple filters combined."""
        filters = FilterState(organism="human", assay_type="ChIP-seq")
        result = manager.apply_filters(sample_df_for_filtering, filters)
        assert len(result) == 1
        assert result.iloc[0]["organism"] == "human"
        assert result.iloc[0]["assay_term_name"] == "ChIP-seq"

    def test_apply_filters_no_match(
        self, manager: SearchFilterManager, sample_df_for_filtering: pd.DataFrame
    ) -> None:
        """Test apply_filters with no matching results."""
        filters = FilterState(organism="worm")
        result = manager.apply_filters(sample_df_for_filtering, filters)
        assert len(result) == 0

    def test_apply_filters_description_search_assay_column(
        self, manager: SearchFilterManager, sample_df_for_filtering: pd.DataFrame
    ) -> None:
        """Test description search matches assay_term_name column."""
        filters = FilterState(description_search="ATAC-seq")
        result = manager.apply_filters(sample_df_for_filtering, filters)
        assert len(result) == 1
        assert result.iloc[0]["assay_term_name"] == "ATAC-seq"

    def test_apply_filters_description_search_biosample_column(
        self, manager: SearchFilterManager, sample_df_for_filtering: pd.DataFrame
    ) -> None:
        """Test description search matches biosample_term_name column."""
        # Search for K562 which appears in biosample_term_name
        filters = FilterState(description_search="K562")
        result = manager.apply_filters(sample_df_for_filtering, filters)
        assert len(result) == 1
        assert result.iloc[0]["biosample_term_name"] == "K562"

    def test_apply_filters_description_search_organism_column(
        self, manager: SearchFilterManager, sample_df_for_filtering: pd.DataFrame
    ) -> None:
        """Test description search matches organism column."""
        # "mouse" appears in organism column AND in description
        filters = FilterState(description_search="mouse")
        result = manager.apply_filters(sample_df_for_filtering, filters)
        # Matches: ENC002 (mouse), ENC003 (description), ENC004 (mouse)
        assert len(result) == 3
        # Verify that it matched rows by organism or description containing "mouse"
        for _, row in result.iterrows():
            has_mouse = (
                "mouse" in str(row.get("organism", "")).lower()
                or "mouse" in str(row.get("description", "")).lower()
            )
            assert has_mouse

    def test_apply_filters_description_search_fuzzy_match(
        self, manager: SearchFilterManager, sample_df_for_filtering: pd.DataFrame
    ) -> None:
        """Test description search with typo uses fuzzy matching."""
        # "cerebelum" is a typo for "cerebellum"
        filters = FilterState(description_search="cerebelum")
        result = manager.apply_filters(sample_df_for_filtering, filters)
        # Should match both cerebellum (biosample_term_name or description)
        assert len(result) >= 1
        # Verify at least one matched row has cerebellum
        has_cerebellum = any(
            "cerebellum" in str(row.get("biosample_term_name", "")).lower()
            or "cerebellum" in str(row.get("description", "")).lower()
            for _, row in result.iterrows()
        )
        assert has_cerebellum


class TestSearchFilterManagerApplyFiltersEdgeCases:
    """Edge case tests for apply_filters method."""

    @pytest.fixture
    def manager(self) -> SearchFilterManager:
        """Create SearchFilterManager instance."""
        return SearchFilterManager()

    def test_apply_filters_missing_organism_column(
        self, manager: SearchFilterManager
    ) -> None:
        """Test apply_filters when organism column is missing."""
        df = pd.DataFrame({"accession": ["ENC001"], "assay_term_name": ["ChIP-seq"]})
        filters = FilterState(organism="human")
        result = manager.apply_filters(df, filters)
        # Should not filter since column is missing
        assert len(result) == 1

    def test_apply_filters_missing_assay_column(
        self, manager: SearchFilterManager
    ) -> None:
        """Test apply_filters when assay_term_name column is missing."""
        df = pd.DataFrame({"accession": ["ENC001"], "organism": ["human"]})
        filters = FilterState(assay_type="ChIP-seq")
        result = manager.apply_filters(df, filters)
        # Should not filter since column is missing
        assert len(result) == 1

    def test_apply_filters_hic_lowercase(self, manager: SearchFilterManager) -> None:
        """Test apply_filters with lowercase hic filter."""
        df = pd.DataFrame(
            {
                "accession": ["ENC001", "ENC002"],
                "assay_term_name": ["HiC", "in situ Hi-C"],
            }
        )
        filters = FilterState(assay_type="hic")
        result = manager.apply_filters(df, filters)
        assert len(result) == 2

    def test_apply_filters_with_combined_text_column(
        self, manager: SearchFilterManager
    ) -> None:
        """Test apply_filters using combined_text column for search."""
        df = pd.DataFrame(
            {
                "accession": ["ENC001"],
                "combined_text": ["chip seq h3k27ac k562 human"],
            }
        )
        filters = FilterState(target="H3K27ac")
        result = manager.apply_filters(df, filters)
        assert len(result) == 1

    def test_apply_filters_with_title_column(
        self, manager: SearchFilterManager
    ) -> None:
        """Test apply_filters using title column for search."""
        df = pd.DataFrame(
            {
                "accession": ["ENC001"],
                "title": ["H3K27ac ChIP-seq on K562"],
            }
        )
        filters = FilterState(target="H3K27ac")
        result = manager.apply_filters(df, filters)
        assert len(result) == 1


# =============================================================================
# Build Search Query Tests
# =============================================================================


class TestSearchFilterManagerBuildSearchQuery:
    """Tests for SearchFilterManager.build_search_query() method."""

    @pytest.fixture
    def manager(self) -> SearchFilterManager:
        """Create SearchFilterManager instance."""
        return SearchFilterManager()

    def test_build_search_query_empty(self, manager: SearchFilterManager) -> None:
        """Test build_search_query with no filters."""
        filters = FilterState()
        query = manager.build_search_query(filters)
        assert query == ""

    def test_build_search_query_assay_only(self, manager: SearchFilterManager) -> None:
        """Test build_search_query with assay_type only."""
        filters = FilterState(assay_type="ChIP-seq")
        query = manager.build_search_query(filters)
        assert query == "ChIP-seq"

    def test_build_search_query_multiple_filters(
        self, manager: SearchFilterManager
    ) -> None:
        """Test build_search_query with multiple filters."""
        filters = FilterState(
            assay_type="ChIP-seq",
            organism="mouse",
            target="H3K27ac",
            biosample="cerebellum",
        )
        query = manager.build_search_query(filters)
        assert "ChIP-seq" in query
        assert "mouse" in query
        assert "H3K27ac" in query
        assert "cerebellum" in query

    def test_build_search_query_with_age_stage(
        self, manager: SearchFilterManager
    ) -> None:
        """Test build_search_query with age_stage."""
        filters = FilterState(age_stage="P60")
        query = manager.build_search_query(filters)
        assert "P60" in query

    def test_build_search_query_with_description_search(
        self, manager: SearchFilterManager
    ) -> None:
        """Test build_search_query with description_search."""
        filters = FilterState(description_search="enhancer analysis")
        query = manager.build_search_query(filters)
        assert "enhancer analysis" in query


# =============================================================================
# parse_age_from_text Tests
# =============================================================================


class TestParseAgeFromText:
    """Tests for parse_age_from_text function."""

    def test_parse_age_empty_text(self) -> None:
        """Test parse_age_from_text with empty text."""
        result = parse_age_from_text("")
        assert result is None

    def test_parse_age_none_text(self) -> None:
        """Test parse_age_from_text with None text."""
        result = parse_age_from_text(None)
        assert result is None

    def test_parse_age_postnatal_p0(self) -> None:
        """Test parse_age_from_text with P0."""
        result = parse_age_from_text("ChIP-seq on P0 mouse brain")
        assert result == "P0"

    def test_parse_age_postnatal_p60(self) -> None:
        """Test parse_age_from_text with P60."""
        result = parse_age_from_text("ATAC-seq on P60 cerebellum")
        assert result == "P60"

    def test_parse_age_embryonic_e14(self) -> None:
        """Test parse_age_from_text with E14.5."""
        result = parse_age_from_text("RNA-seq of E14.5 mouse forebrain")
        assert result == "E14.5"

    def test_parse_age_weeks(self) -> None:
        """Test parse_age_from_text with weeks."""
        result = parse_age_from_text("ChIP-seq on 8 week mouse cerebellum")
        assert result == "8 weeks"

    def test_parse_age_weeks_hyphenated(self) -> None:
        """Test parse_age_from_text with hyphenated weeks."""
        result = parse_age_from_text("ChIP-seq on 8-week mouse cerebellum")
        assert result == "8 weeks"

    def test_parse_age_months(self) -> None:
        """Test parse_age_from_text with months."""
        result = parse_age_from_text("RNA-seq of 3 month mouse liver")
        assert result == "3 months"

    def test_parse_age_months_hyphenated(self) -> None:
        """Test parse_age_from_text with hyphenated months."""
        result = parse_age_from_text("ATAC-seq on 2-month old mice")
        assert result == "2 months"

    def test_parse_age_adult_term(self) -> None:
        """Test parse_age_from_text with 'adult' term."""
        result = parse_age_from_text("ChIP-seq on adult mouse brain")
        assert result == "adult"

    def test_parse_age_newborn_term(self) -> None:
        """Test parse_age_from_text with 'newborn' term."""
        result = parse_age_from_text("RNA-seq of newborn mouse heart")
        assert result == "newborn"

    def test_parse_age_embryonic_term(self) -> None:
        """Test parse_age_from_text with 'embryonic' term."""
        result = parse_age_from_text("ChIP-seq on embryonic stem cells")
        assert result == "embryonic"

    def test_parse_age_fetal_term(self) -> None:
        """Test parse_age_from_text with 'fetal' term."""
        result = parse_age_from_text("RNA-seq of fetal liver")
        assert result == "fetal"

    def test_parse_age_no_age_found(self) -> None:
        """Test parse_age_from_text with no age information."""
        result = parse_age_from_text("ChIP-seq on K562 cells targeting H3K27ac")
        assert result is None

    def test_parse_age_lowercase_embryonic(self) -> None:
        """Test parse_age_from_text handles lowercase."""
        result = parse_age_from_text("rna-seq e10.5 mouse")
        assert result == "E10.5"

    def test_parse_age_juvenile_term(self) -> None:
        """Test parse_age_from_text with 'juvenile' term."""
        result = parse_age_from_text("ATAC-seq on juvenile mouse brain")
        assert result == "juvenile"

    def test_parse_age_aged_term(self) -> None:
        """Test parse_age_from_text with 'aged' term."""
        result = parse_age_from_text("RNA-seq of aged mouse liver")
        assert result == "aged"


# =============================================================================
# Coverage Gap Tests
# =============================================================================


class TestSearchFilterManagerFuzzySearchNoColumns:
    """Tests for _fuzzy_text_search with no available columns (line 535)."""

    @pytest.fixture
    def manager(self) -> SearchFilterManager:
        """Create SearchFilterManager instance."""
        return SearchFilterManager()

    def test_fuzzy_text_search_no_searchable_columns(
        self, manager: SearchFilterManager
    ) -> None:
        """Line 535: _fuzzy_text_search returns all True when no searchable columns."""
        # Create DataFrame with only non-searchable columns
        df = pd.DataFrame(
            {
                "accession": ["ENC001", "ENC002", "ENC003"],
                "file_count": [10, 20, 30],
                "replicate_count": [2, 3, 4],
            }
        )

        # Call _fuzzy_text_search directly (private method)
        result = manager._fuzzy_text_search(df, ["some", "search", "terms"])

        # Should return all True since no searchable columns exist
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert result.all()  # All rows should be True


class TestSearchFilterManagerApplyFiltersBodyPart:
    """Tests for apply_filters with body_part filter (lines 627-632)."""

    @pytest.fixture
    def manager(self) -> SearchFilterManager:
        """Create SearchFilterManager instance."""
        return SearchFilterManager()

    def test_apply_filters_body_part_filter(self, manager: SearchFilterManager) -> None:
        """Lines 627-632: Body part filter calls get_biosamples_for_organ."""
        # Create DataFrame with brain biosamples and non-brain biosamples
        df = pd.DataFrame(
            {
                "accession": ["ENC001", "ENC002", "ENC003", "ENC004"],
                "biosample_term_name": ["cerebellum", "forebrain", "liver", "K562"],
                "organism": ["mouse", "mouse", "human", "human"],
            }
        )

        # Filter by body_part = "brain"
        filters = FilterState(body_part="brain")
        result = manager.apply_filters(df, filters)

        # Should only include brain biosamples (cerebellum, forebrain)
        assert len(result) >= 1
        # Verify no liver or K562 in results
        biosamples = result["biosample_term_name"].tolist()
        assert "liver" not in biosamples
        assert "K562" not in biosamples

    def test_apply_filters_body_part_with_no_matches(
        self, manager: SearchFilterManager
    ) -> None:
        """Test body_part filter when no biosamples match the organ."""
        df = pd.DataFrame(
            {
                "accession": ["ENC001", "ENC002"],
                "biosample_term_name": ["K562", "HepG2"],
                "organism": ["human", "human"],
            }
        )

        # Filter by body_part = "brain" - neither K562 nor HepG2 are brain tissues
        filters = FilterState(body_part="brain")
        result = manager.apply_filters(df, filters)

        # Should return no results since K562 and HepG2 are not brain biosamples
        assert len(result) == 0

    def test_apply_filters_body_part_blood(self, manager: SearchFilterManager) -> None:
        """Test body_part filter for blood/bodily fluid organ."""
        df = pd.DataFrame(
            {
                "accession": ["ENC001", "ENC002", "ENC003"],
                "biosample_term_name": ["K562", "GM12878", "cerebellum"],
                "organism": ["human", "human", "mouse"],
            }
        )

        # Filter by body_part that includes cell lines - use "blood" or similar
        # K562 and GM12878 are blood-related cell lines
        filters = FilterState(body_part="blood")
        result = manager.apply_filters(df, filters)

        # Should include blood-related samples but not brain
        biosamples = result["biosample_term_name"].tolist()
        assert "cerebellum" not in biosamples
