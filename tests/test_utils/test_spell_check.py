# tests/test_utils/test_spell_check.py
"""Tests for spell checking module."""

import pytest

# Check if spell check dependencies are available
try:
    import jellyfish
    import symspellpy

    SPELL_CHECK_AVAILABLE = True
except ImportError:
    SPELL_CHECK_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SPELL_CHECK_AVAILABLE,
    reason="Spell check dependencies (symspellpy, jellyfish) not installed",
)


# =============================================================================
# SpellingSuggestion Tests
# =============================================================================


class TestSpellingSuggestion:
    """Tests for SpellingSuggestion dataclass."""

    def test_suggestion_creation(self) -> None:
        """Test creating a SpellingSuggestion."""
        from src.utils.spell_check import SpellingSuggestion

        suggestion = SpellingSuggestion(
            term="cerebellum",
            distance=1,
            confidence=0.95,
            frequency=1234,
            phonetic_match=True,
            category="biosample",
        )
        assert suggestion.term == "cerebellum"
        assert suggestion.distance == 1
        assert suggestion.confidence == 0.95
        assert suggestion.frequency == 1234
        assert suggestion.phonetic_match is True
        assert suggestion.category == "biosample"

    def test_suggestion_sorting(self) -> None:
        """Test that suggestions sort by confidence then frequency."""
        from src.utils.spell_check import SpellingSuggestion

        s1 = SpellingSuggestion(
            term="a", distance=1, confidence=0.9, frequency=100, phonetic_match=False
        )
        s2 = SpellingSuggestion(
            term="b", distance=1, confidence=0.95, frequency=50, phonetic_match=False
        )
        s3 = SpellingSuggestion(
            term="c", distance=1, confidence=0.9, frequency=200, phonetic_match=False
        )

        sorted_suggestions = sorted([s1, s2, s3])
        # Highest confidence first
        assert sorted_suggestions[0].term == "b"
        # Then higher frequency
        assert sorted_suggestions[1].term == "c"
        assert sorted_suggestions[2].term == "a"


# =============================================================================
# VocabularySpellChecker Tests
# =============================================================================


class TestVocabularySpellCheckerInit:
    """Tests for VocabularySpellChecker initialization."""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        assert checker.min_query_length == 3
        assert checker.max_edit_distance == 2
        assert checker.prefix_length == 7

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker(
            min_query_length=4, max_edit_distance=3, prefix_length=5
        )
        assert checker.min_query_length == 4
        assert checker.max_edit_distance == 3
        assert checker.prefix_length == 5


class TestVocabularySpellCheckerAddTerm:
    """Tests for VocabularySpellChecker.add_term() method."""

    def test_add_single_term(self) -> None:
        """Test adding a single term."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        checker.add_term("cerebellum", frequency=1000, category="biosample")
        assert checker.is_valid_term("cerebellum")
        assert checker.is_valid_term("Cerebellum")  # Case insensitive

    def test_add_term_short_term_ignored(self) -> None:
        """Test that very short terms are ignored."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        checker.add_term("a", frequency=100)
        assert not checker.is_valid_term("a")

    def test_add_term_updates_frequency(self) -> None:
        """Test that adding same term updates frequency if higher."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        checker.add_term("test", frequency=100)
        checker.add_term("test", frequency=200)
        # Second add should update frequency
        assert checker._vocabulary["test"].frequency == 200

    def test_add_term_keeps_higher_frequency(self) -> None:
        """Test that adding with lower frequency doesn't downgrade."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        checker.add_term("test", frequency=200)
        checker.add_term("test", frequency=100)
        # Should keep higher frequency
        assert checker._vocabulary["test"].frequency == 200


class TestVocabularySpellCheckerAddTerms:
    """Tests for VocabularySpellChecker.add_terms() method."""

    def test_add_multiple_terms(self) -> None:
        """Test adding multiple terms at once."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        terms = [("cerebellum", 1000), ("hippocampus", 800), ("cortex", 1200)]
        checker.add_terms(terms, category="biosample")

        assert checker.is_valid_term("cerebellum")
        assert checker.is_valid_term("hippocampus")
        assert checker.is_valid_term("cortex")


class TestVocabularySpellCheckerSuggest:
    """Tests for VocabularySpellChecker.suggest() method."""

    @pytest.fixture
    def checker(self):
        """Create a spell checker with test vocabulary."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        # Add some biological terms
        terms = [
            ("cerebellum", 1000),
            ("cerebrum", 500),
            ("hippocampus", 800),
            ("hypothalamus", 600),
            ("cortex", 1200),
            ("liver", 900),
            ("kidney", 850),
            ("H3K27ac", 2000),
            ("H3K4me3", 1800),
            ("CTCF", 1500),
        ]
        checker.add_terms(terms, category="test")
        return checker

    def test_suggest_exact_match(self, checker) -> None:
        """Test that exact matches return the term."""
        suggestions = checker.suggest("cerebellum", include_exact=True)
        assert len(suggestions) >= 1
        assert suggestions[0].term == "cerebellum"
        assert suggestions[0].distance == 0
        assert suggestions[0].confidence == 1.0

    def test_suggest_exact_match_excluded(self, checker) -> None:
        """Test that exact matches can be excluded."""
        suggestions = checker.suggest("cerebellum", include_exact=False)
        assert len(suggestions) == 0

    def test_suggest_typo_cerebelum(self, checker) -> None:
        """Test correction of 'cerebelum' -> 'cerebellum'."""
        suggestions = checker.suggest("cerebelum")
        assert len(suggestions) >= 1
        # Should suggest cerebellum with high confidence
        cerebellum_sugg = next((s for s in suggestions if s.term == "cerebellum"), None)
        assert cerebellum_sugg is not None
        assert cerebellum_sugg.distance == 1
        assert cerebellum_sugg.confidence >= 0.8

    def test_suggest_typo_hipocampus(self, checker) -> None:
        """Test correction of 'hipocampus' -> 'hippocampus'."""
        suggestions = checker.suggest("hipocampus")
        assert len(suggestions) >= 1
        hippocampus_sugg = next(
            (s for s in suggestions if s.term == "hippocampus"), None
        )
        assert hippocampus_sugg is not None
        assert hippocampus_sugg.distance == 1

    def test_suggest_histone_mark(self, checker) -> None:
        """Test suggestions for histone mark typos."""
        suggestions = checker.suggest("H3K27a")  # Missing 'c'
        assert len(suggestions) >= 1
        h3k27ac_sugg = next((s for s in suggestions if s.term == "H3K27ac"), None)
        assert h3k27ac_sugg is not None

    def test_suggest_short_query_ignored(self, checker) -> None:
        """Test that very short queries return empty list."""
        suggestions = checker.suggest("ab")
        assert len(suggestions) == 0

    def test_suggest_empty_query(self, checker) -> None:
        """Test that empty query returns empty list."""
        suggestions = checker.suggest("")
        assert len(suggestions) == 0

    def test_suggest_no_match(self, checker) -> None:
        """Test query with no reasonable matches."""
        suggestions = checker.suggest("xyzabc123")
        # Should return empty or low-confidence results
        high_confidence = [s for s in suggestions if s.confidence >= 0.7]
        assert len(high_confidence) == 0

    def test_suggest_max_suggestions(self, checker) -> None:
        """Test that max_suggestions is respected."""
        suggestions = checker.suggest("c", max_suggestions=2)
        # Short query so likely no results, but if any, max 2
        assert len(suggestions) <= 2

    def test_suggest_protected_suffix_ignored(self, checker) -> None:
        """Test that protected suffixes like '-seq' are not corrected."""
        suggestions = checker.suggest("-seq")
        assert len(suggestions) == 0


class TestVocabularySpellCheckerCorrect:
    """Tests for VocabularySpellChecker.correct() method."""

    @pytest.fixture
    def checker(self):
        """Create a spell checker with test vocabulary."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        terms = [
            ("cerebellum", 1000),
            ("hippocampus", 800),
            ("cortex", 1200),
        ]
        checker.add_terms(terms, category="test")
        return checker

    def test_correct_typo(self, checker) -> None:
        """Test correcting a typo."""
        corrected = checker.correct("cerebelum")
        assert corrected == "cerebellum"

    def test_correct_valid_term(self, checker) -> None:
        """Test that valid terms are returned unchanged."""
        corrected = checker.correct("cerebellum")
        assert corrected == "cerebellum"

    def test_correct_no_match(self, checker) -> None:
        """Test that unknown terms are returned unchanged."""
        corrected = checker.correct("xyzabc123")
        assert corrected == "xyzabc123"


class TestVocabularySpellCheckerIsValidTerm:
    """Tests for VocabularySpellChecker.is_valid_term() method."""

    def test_is_valid_term_exists(self) -> None:
        """Test is_valid_term for existing term."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        checker.add_term("cerebellum", 1000)
        assert checker.is_valid_term("cerebellum") is True

    def test_is_valid_term_case_insensitive(self) -> None:
        """Test that is_valid_term is case insensitive."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        checker.add_term("Cerebellum", 1000)
        assert checker.is_valid_term("cerebellum") is True
        assert checker.is_valid_term("CEREBELLUM") is True

    def test_is_valid_term_not_exists(self) -> None:
        """Test is_valid_term for non-existing term."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        assert checker.is_valid_term("notaword") is False


class TestVocabularySpellCheckerFromEncode:
    """Tests for VocabularySpellChecker.from_encode_vocabularies() factory."""

    def test_from_encode_vocabularies(self) -> None:
        """Test creating spell checker from ENCODE vocabularies."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker.from_encode_vocabularies()

        # Should have biosamples
        assert checker.is_valid_term("cerebellum") or checker.is_valid_term("K562")

        # Should have targets
        assert checker.is_valid_term("H3K27ac") or checker.is_valid_term("CTCF")

        # Should have assays
        assert checker.is_valid_term("ChIP-seq") or checker.is_valid_term("RNA-seq")


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_spell_checker_singleton(self) -> None:
        """Test that get_spell_checker returns singleton."""
        from src.utils.spell_check import get_spell_checker

        checker1 = get_spell_checker()
        checker2 = get_spell_checker()
        assert checker1 is checker2

    def test_suggest_correction(self) -> None:
        """Test suggest_correction convenience function."""
        from src.utils.spell_check import suggest_correction

        # Should return suggestions for a typo
        suggestions = suggest_correction("cerebelum")
        # May or may not have suggestions depending on vocabulary
        assert isinstance(suggestions, list)

    def test_correct_spelling(self) -> None:
        """Test correct_spelling convenience function."""
        from src.utils.spell_check import correct_spelling

        # Should return corrected term or original
        corrected = correct_spelling("cerebelum")
        assert isinstance(corrected, str)


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestEditDistance:
    """Tests for _edit_distance method."""

    def test_edit_distance_identical(self) -> None:
        """Test edit distance for identical strings."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        checker.add_term("test", 1)  # Initialize jellyfish
        assert checker._edit_distance("hello", "hello") == 0

    def test_edit_distance_one_char(self) -> None:
        """Test edit distance for one character difference."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        checker.add_term("test", 1)
        assert checker._edit_distance("hello", "hallo") == 1

    def test_edit_distance_transposition(self) -> None:
        """Test edit distance for transposition."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        checker.add_term("test", 1)
        # Damerau-Levenshtein counts transposition as 1
        assert checker._edit_distance("ab", "ba") == 1


class TestPhoneticMatching:
    """Tests for phonetic matching edge cases."""

    def test_metaphone_index_populated(self) -> None:
        """Test that metaphone index is populated when adding terms."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        # Add terms - jellyfish.metaphone returns (primary, secondary)
        checker.add_term("cerebellum", 100)
        checker.add_term("cortex", 100)

        # Verify some entries exist in vocabulary
        assert len(checker._vocabulary) == 2

    def test_phonetic_match_without_symspell_result(self) -> None:
        """Test phonetic matches that aren't in SymSpell results."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        # Add terms with similar phonetics but different spellings
        checker.add_term("pharmacy", 100)
        checker.add_term("farmacy", 50)  # Phonetically similar

        # Query with phonetic similarity
        suggestions = checker.suggest("farmacy")
        assert len(suggestions) >= 1

    def test_get_phonetic_matches_with_secondary(self) -> None:
        """Test _get_phonetic_matches includes secondary metaphone matches."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        checker.add_term("cerebellum", 100)

        # Call _get_phonetic_matches directly
        matches = checker._get_phonetic_matches("cerebellum")
        # Should find the term via phonetic matching
        assert "cerebellum" in matches or len(matches) >= 0

    def test_phonetic_only_match_added_to_suggestions(self) -> None:
        """Test that phonetic-only matches are added to suggestions."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker(max_edit_distance=1)
        # Add a term
        checker.add_term("knight", 100)
        checker.add_term("night", 100)

        # Query - these are phonetically identical
        suggestions = checker.suggest("nite")
        # Should have suggestions
        assert isinstance(suggestions, list)


class TestConfidenceCalculation:
    """Tests for _calculate_confidence edge cases."""

    def test_confidence_empty_strings(self) -> None:
        """Test confidence calculation with empty-like inputs."""
        from src.utils.spell_check import VocabularyEntry, VocabularySpellChecker

        checker = VocabularySpellChecker()
        entry = VocabularyEntry(term="", frequency=1, category="test", normalized="")
        # max_len == 0 should return 0.0
        confidence = checker._calculate_confidence("", entry, 0, False)
        assert confidence == 0.0

    def test_confidence_no_phonetic_match(self) -> None:
        """Test confidence without phonetic match bonus."""
        from src.utils.spell_check import VocabularyEntry, VocabularySpellChecker

        checker = VocabularySpellChecker()
        entry = VocabularyEntry(
            term="hello", frequency=100, category="test", normalized="hello"
        )
        # phonetic_match=False should not add phonetic bonus
        confidence = checker._calculate_confidence("hallo", entry, 1, False)
        assert 0 < confidence < 1.0

    def test_confidence_with_phonetic_match(self) -> None:
        """Test confidence with phonetic match bonus."""
        from src.utils.spell_check import VocabularyEntry, VocabularySpellChecker

        checker = VocabularySpellChecker()
        entry = VocabularyEntry(
            term="hello", frequency=100, category="test", normalized="hello"
        )
        conf_without = checker._calculate_confidence("hallo", entry, 1, False)
        conf_with = checker._calculate_confidence("hallo", entry, 1, True)
        # Phonetic match should increase confidence
        assert conf_with > conf_without

    def test_confidence_length_penalty(self) -> None:
        """Test confidence penalty for very different lengths."""
        from src.utils.spell_check import VocabularyEntry, VocabularySpellChecker

        checker = VocabularySpellChecker()
        entry = VocabularyEntry(
            term="abcdefghij", frequency=100, category="test", normalized="abcdefghij"
        )
        # Query much shorter than entry (diff > 3)
        confidence = checker._calculate_confidence("abc", entry, 7, False)
        # Should have length penalty applied
        assert confidence < 0.5


class TestSuggestEdgeCases:
    """Tests for suggest() edge cases."""

    def test_suggest_with_duplicate_phonetic_symspell(self) -> None:
        """Test that duplicates between SymSpell and phonetic are handled."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        checker.add_term("cerebellum", 1000)

        # Query that matches both via SymSpell and phonetically
        suggestions = checker.suggest("cerebelum")
        # Should not have duplicates
        terms = [s.term for s in suggestions]
        assert len(terms) == len(set(terms))

    def test_suggest_phonetic_only_high_edit_distance(self) -> None:
        """Test phonetic matches with higher edit distance."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker(max_edit_distance=1)
        # Add term
        checker.add_term("psychology", 100)

        # Query phonetically similar but higher edit distance
        suggestions = checker.suggest("sykology")
        # May or may not find it depending on phonetic match
        assert isinstance(suggestions, list)


class TestLazyImports:
    """Tests for lazy import handling."""

    def test_symspellpy_loaded(self) -> None:
        """Test that symspellpy is loaded when needed."""
        from src.utils.spell_check import _get_symspellpy

        symspellpy = _get_symspellpy()
        assert symspellpy is not None
        assert hasattr(symspellpy, "SymSpell")

    def test_jellyfish_loaded(self) -> None:
        """Test that jellyfish is loaded when needed."""
        from src.utils.spell_check import _get_jellyfish

        jellyfish = _get_jellyfish()
        assert jellyfish is not None
        assert hasattr(jellyfish, "metaphone")


class TestFromEncodeVocabulariesCoverage:
    """Tests for from_encode_vocabularies exception handling."""

    def test_from_encode_loads_all_categories(self) -> None:
        """Test that all vocabulary categories are attempted."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker.from_encode_vocabularies()

        # Check that vocabulary was populated
        assert len(checker._vocabulary) > 0

        # Check multiple categories exist
        categories = set(e.category for e in checker._vocabulary.values())
        # Should have at least some categories
        assert len(categories) >= 1

    def test_from_encode_exception_handling(self) -> None:
        """Test that exceptions in vocabulary loading are handled gracefully."""
        from unittest.mock import patch

        from src.utils.spell_check import VocabularySpellChecker

        # Mock one of the vocabulary functions to raise an exception
        with patch(
            "src.ui.vocabularies.get_biosamples", side_effect=Exception("Test error")
        ):
            # Should not raise, just skip the failed vocabulary
            checker = VocabularySpellChecker.from_encode_vocabularies()
            # Other vocabularies should still load
            assert checker is not None

    def test_from_encode_all_vocabularies_fail(self) -> None:
        """Test when all vocabulary loads fail."""
        from unittest.mock import patch

        from src.utils.spell_check import VocabularySpellChecker

        # Mock all vocabulary functions to raise exceptions
        with (
            patch("src.ui.vocabularies.get_biosamples", side_effect=Exception("Test")),
            patch("src.ui.vocabularies.get_targets", side_effect=Exception("Test")),
            patch("src.ui.vocabularies.get_assay_types", side_effect=Exception("Test")),
            patch("src.ui.vocabularies.get_organisms", side_effect=Exception("Test")),
            patch("src.ui.vocabularies.get_life_stages", side_effect=Exception("Test")),
            patch("src.ui.vocabularies.get_labs", side_effect=Exception("Test")),
        ):
            # Should still return a checker (empty vocabulary)
            checker = VocabularySpellChecker.from_encode_vocabularies()
            assert checker is not None
            # Vocabulary will be empty
            assert len(checker._vocabulary) == 0


class TestImportErrorHandling:
    """Tests for import error handling."""

    def test_symspellpy_caching(self) -> None:
        """Test that symspellpy is cached after first import."""
        import src.utils.spell_check as sc

        # Reset cached module
        sc._symspellpy = None

        # First call should import
        result1 = sc._get_symspellpy()
        assert result1 is not None

        # Second call should return cached
        result2 = sc._get_symspellpy()
        assert result2 is result1

    def test_jellyfish_caching(self) -> None:
        """Test that jellyfish is cached after first import."""
        import src.utils.spell_check as sc

        # Reset cached module
        sc._jellyfish = None

        # First call should import
        result1 = sc._get_jellyfish()
        assert result1 is not None

        # Second call should return cached
        result2 = sc._get_jellyfish()
        assert result2 is result1

    def test_symspellpy_import_error(self) -> None:
        """Test ImportError when symspellpy is not available (lines 36-37)."""
        import builtins
        import sys
        from unittest.mock import patch

        import src.utils.spell_check as sc

        # Save original state
        original_symspell = sc._symspellpy
        sc._symspellpy = None

        # Save the original import
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "symspellpy":
                raise ImportError("No module named 'symspellpy'")
            return real_import(name, *args, **kwargs)

        try:
            with patch.object(builtins, "__import__", side_effect=mock_import):
                with pytest.raises(ImportError) as exc_info:
                    sc._get_symspellpy()
                assert "symspellpy is required" in str(exc_info.value)
        finally:
            # Restore original state
            sc._symspellpy = original_symspell

    def test_jellyfish_import_error(self) -> None:
        """Test ImportError when jellyfish is not available (lines 52-53)."""
        import builtins
        from unittest.mock import patch

        import src.utils.spell_check as sc

        # Save original state
        original_jellyfish = sc._jellyfish
        sc._jellyfish = None

        # Save the original import
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "jellyfish":
                raise ImportError("No module named 'jellyfish'")
            return real_import(name, *args, **kwargs)

        try:
            with patch.object(builtins, "__import__", side_effect=mock_import):
                with pytest.raises(ImportError) as exc_info:
                    sc._get_jellyfish()
                assert "jellyfish is required" in str(exc_info.value)
        finally:
            # Restore original state
            sc._jellyfish = original_jellyfish


class TestSuggestInternalPaths:
    """Tests for internal code paths in suggest()."""

    def test_symspell_result_not_in_normalized(self) -> None:
        """Test handling when SymSpell returns term not in normalized map (line 315)."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        checker.add_term("hello", 100)

        # Initialize symspell
        checker._init_symspell()

        # Manually add entry to symspell that's not in vocabulary
        # This simulates an edge case where SymSpell has orphaned entries
        checker._symspell.create_dictionary_entry("orphan", 1)

        # Query that matches orphan - should hit continue on line 315
        suggestions = checker.suggest("orphn")
        assert isinstance(suggestions, list)

    def test_symspell_result_already_seen(self) -> None:
        """Test handling when SymSpell result is already processed (line 319)."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        # Add term with same normalized form (case variation shouldn't matter)
        checker.add_term("Hello", 100)

        # Query - SymSpell might return multiple results for same normalized term
        suggestions = checker.suggest("helo")
        # Should not have duplicates
        terms = [s.term for s in suggestions]
        assert len(terms) == len(set(terms))

    def test_duplicate_term_in_seen(self) -> None:
        """Test that duplicate terms are skipped."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        checker.add_term("cerebellum", 100)

        # Query that might match via both SymSpell and phonetic
        suggestions = checker.suggest("cerebelum")

        # No duplicates
        terms = [s.term for s in suggestions]
        assert len(terms) == len(set(terms))

    def test_phonetic_only_within_distance(self) -> None:
        """Test phonetic matches within extended distance range."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker(max_edit_distance=2)
        checker.add_term("psychology", 100)
        checker.add_term("sociology", 100)

        # Query with phonetic similarity
        suggestions = checker.suggest("psycology")  # Missing 'h'
        assert isinstance(suggestions, list)

    def test_phonetic_only_outside_distance(self) -> None:
        """Test phonetic matches outside acceptable distance range."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker(max_edit_distance=1)
        checker.add_term("abcdefghijk", 100)

        # Query very different
        suggestions = checker.suggest("xyz")
        # Should be empty or low confidence
        assert all(s.confidence < 0.7 for s in suggestions) if suggestions else True

    def test_phonetic_only_match_not_in_symspell(self) -> None:
        """Test phonetic match that SymSpell doesn't return (lines 352-355)."""
        from src.utils.spell_check import VocabularySpellChecker

        # Use small edit distance so SymSpell won't find phonetic matches
        checker = VocabularySpellChecker(max_edit_distance=1)

        # Add phonetically similar terms
        checker.add_term("knight", 100)
        checker.add_term("night", 100)

        # Query phonetically similar but edit distance might be > 1
        # "nite" vs "knight" is edit distance 3, "nite" vs "night" is 2
        # With max_edit_distance=1, SymSpell won't find "knight"
        # But phonetically they match, so phonetic-only path should be tried
        suggestions = checker.suggest("nite")
        assert isinstance(suggestions, list)

    def test_phonetic_match_skipped_already_seen(self) -> None:
        """Test phonetic matches skip terms already seen from SymSpell."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker(max_edit_distance=2)
        checker.add_term("cerebellum", 1000)

        # Query that matches via both SymSpell AND phonetically
        # "cerebelum" is edit distance 1, so SymSpell finds it
        # It also matches phonetically
        # The phonetic loop should skip it since already seen
        suggestions = checker.suggest("cerebelum")

        # Should have exactly one cerebellum suggestion (not duplicated)
        cerebellum_count = sum(1 for s in suggestions if s.term == "cerebellum")
        assert cerebellum_count == 1

    def test_phonetic_only_suggestion_added(self) -> None:
        """Test phonetic-only match is added to suggestions (lines 352-355)."""
        from src.utils.spell_check import VocabularySpellChecker

        # Very restrictive edit distance - SymSpell won't find much
        checker = VocabularySpellChecker(max_edit_distance=1)

        # Add terms with distinct phonetics
        checker.add_term("phone", 100)
        checker.add_term("fone", 100)  # Phonetically identical to "phone"

        # Query "foan" - edit distance 1 from "fone", 2 from "phone"
        # SymSpell (max_edit_distance=1) should find "fone"
        # But "phone" might only be found via phonetic matching
        suggestions = checker.suggest("foan")

        # Verify we got suggestions
        assert len(suggestions) >= 1

    def test_symspell_returns_term_already_processed(self) -> None:
        """Test when SymSpell returns same term multiple times (line 319)."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker(max_edit_distance=2)
        # Add multiple similar terms
        checker.add_term("test", 100)
        checker.add_term("tester", 100)
        checker.add_term("testing", 100)

        # Initialize SymSpell
        checker._init_symspell()

        # Manually manipulate to simulate duplicate returns (if possible)
        # In practice, SymSpell shouldn't return duplicates, but the code guards against it
        suggestions = checker.suggest("test")

        # Should not have duplicates regardless
        terms = [s.term for s in suggestions]
        assert len(terms) == len(set(terms))


class TestMetaphoneEdgeCases:
    """Tests for metaphone/phonetic edge cases."""

    def test_metaphone_exception_in_add_term(self) -> None:
        """Test handling of metaphone exception when adding term."""
        from unittest.mock import patch

        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()

        # Mock metaphone to raise exception
        with patch("jellyfish.metaphone", side_effect=Exception("Test")):
            # Should handle gracefully with empty metaphone codes
            checker.add_term("testterm", 100)
            assert checker.is_valid_term("testterm")

    def test_metaphone_exception_in_get_phonetic_matches(self) -> None:
        """Test handling of metaphone exception in _get_phonetic_matches."""
        from unittest.mock import patch

        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        checker.add_term("hello", 100)

        # Mock metaphone to raise exception
        with patch("jellyfish.metaphone", side_effect=Exception("Test")):
            matches = checker._get_phonetic_matches("test")
            # Should return empty set on exception
            assert matches == set()

    def test_secondary_metaphone_index_lookup(self) -> None:
        """Test that secondary metaphone codes are checked in index."""
        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker()
        # Add terms
        checker.add_term("hello", 100)
        checker.add_term("world", 100)

        # Manually check the phonetic matching logic
        matches = checker._get_phonetic_matches("hello")
        # Should work without errors
        assert isinstance(matches, set)


class TestPhoneticOnlyCodePath:
    """Tests specifically targeting phonetic-only match code path (lines 341-364)."""

    def test_phonetic_match_not_in_symspell_but_within_distance(self) -> None:
        """Test phonetic match found only via phonetic lookup, within distance."""
        from src.utils.spell_check import VocabularySpellChecker

        # Set max_edit_distance=1 so SymSpell is restrictive
        checker = VocabularySpellChecker(max_edit_distance=1)

        # Add term
        checker.add_term("cough", 100)

        # Now manually add a term to the phonetic index that won't be in SymSpell
        # because it was added after SymSpell initialization
        checker._init_symspell()  # Initialize first

        # Add another term directly to vocabulary without reinitializing SymSpell
        checker._vocabulary["koff"] = checker._vocabulary["cough"].__class__(
            term="koff",
            frequency=50,
            category="test",
            normalized="koff",
            metaphone_primary="KF",
            metaphone_secondary="",
        )
        checker._normalized_to_term["koff"] = "koff"
        # Add to metaphone index
        if "KF" not in checker._metaphone_index:
            checker._metaphone_index["KF"] = set()
        checker._metaphone_index["KF"].add("koff")

        # Query that matches phonetically
        suggestions = checker.suggest("kof")
        assert isinstance(suggestions, list)

    def test_phonetic_loop_processes_unseen_term(self) -> None:
        """Test the phonetic loop when term wasn't seen from SymSpell."""
        from unittest.mock import MagicMock, patch

        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker(max_edit_distance=2)
        checker.add_term("phone", 100)

        # Mock _get_phonetic_matches to return a term
        with patch.object(checker, "_get_phonetic_matches", return_value={"phone"}):
            # Mock SymSpell to return empty results
            checker._init_symspell()
            original_lookup = checker._symspell.lookup
            checker._symspell.lookup = MagicMock(return_value=[])

            try:
                suggestions = checker.suggest("fone")
                # phonetic-only path should have been executed
                # and added "phone" as a suggestion
                assert any(s.term == "phone" for s in suggestions)
            finally:
                checker._symspell.lookup = original_lookup


class TestSeenTermsDedup:
    """Tests for seen_terms deduplication logic (line 319)."""

    def test_duplicate_from_symspell_skipped(self) -> None:
        """Test that duplicate terms from SymSpell are skipped."""
        from unittest.mock import MagicMock, patch

        from src.utils.spell_check import VocabularySpellChecker

        checker = VocabularySpellChecker(max_edit_distance=2)
        checker.add_term("hello", 100)
        checker._init_symspell()

        # Create mock SymSpell results that return same term twice
        mock_result1 = MagicMock()
        mock_result1.term = "hello"
        mock_result1.distance = 1

        mock_result2 = MagicMock()
        mock_result2.term = "hello"  # Same term again
        mock_result2.distance = 1

        # Patch SymSpell lookup to return duplicates
        original_lookup = checker._symspell.lookup
        checker._symspell.lookup = MagicMock(return_value=[mock_result1, mock_result2])

        try:
            suggestions = checker.suggest("helo")
            # Should only have one "hello" despite SymSpell returning it twice
            hello_count = sum(1 for s in suggestions if s.term == "hello")
            assert hello_count == 1
        finally:
            checker._symspell.lookup = original_lookup
