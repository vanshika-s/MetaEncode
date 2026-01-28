# src/utils/spell_check.py
"""Vocabulary-aware spell checking for biological terms.

This module provides spell correction optimized for ENCODE metadata terms,
including biosamples, targets, assays, organisms, and labs. It uses:
- SymSpell for fast edit distance lookups
- Double Metaphone for phonetic matching of scientific terms
- Frequency weighting based on experiment counts

Example:
    >>> checker = VocabularySpellChecker.from_encode_vocabularies()
    >>> checker.suggest("cerebelum")
    [SpellingSuggestion(term='cerebellum', distance=1, confidence=0.95, ...)]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

# Lazy imports for optional dependencies
_symspellpy = None
_jellyfish = None


def _get_symspellpy():
    """Lazy import symspellpy."""
    global _symspellpy
    if _symspellpy is None:
        try:
            import symspellpy

            _symspellpy = symspellpy
        except ImportError:
            raise ImportError(
                "symspellpy is required for spell checking. "
                "Install with: pip install symspellpy"
            )
    return _symspellpy


def _get_jellyfish():
    """Lazy import jellyfish."""
    global _jellyfish
    if _jellyfish is None:
        try:
            import jellyfish

            _jellyfish = jellyfish
        except ImportError:
            raise ImportError(
                "jellyfish is required for phonetic matching. "
                "Install with: pip install jellyfish"
            )
    return _jellyfish


@dataclass
class SpellingSuggestion:
    """A spelling correction suggestion with metadata."""

    term: str
    distance: int  # Edit distance from query
    confidence: float  # 0.0 to 1.0
    frequency: int  # Experiment count (popularity)
    phonetic_match: bool  # True if phonetic codes match
    category: Optional[str] = None  # e.g., "biosample", "target"

    def __lt__(self, other: "SpellingSuggestion") -> bool:
        """Sort by confidence descending, then frequency descending."""
        if self.confidence != other.confidence:
            return self.confidence > other.confidence
        return self.frequency > other.frequency


@dataclass
class VocabularyEntry:
    """An entry in the spell-check vocabulary."""

    term: str
    frequency: int
    category: str
    metaphone_primary: str = ""
    metaphone_secondary: str = ""
    normalized: str = ""  # Lowercase, stripped


class VocabularySpellChecker:
    """Spell checker optimized for ENCODE biological vocabulary.

    Uses a combination of edit distance (SymSpell) and phonetic matching
    (Double Metaphone) to suggest corrections for misspelled terms.

    The vocabulary is built from ENCODE metadata facets, with terms weighted
    by their experiment counts for better ranking.

    Attributes:
        min_query_length: Minimum query length to attempt correction (default: 3)
        max_edit_distance: Maximum edit distance for suggestions (default: 2)
        prefix_length: SymSpell prefix length optimization (default: 7)
    """

    def __init__(
        self,
        min_query_length: int = 3,
        max_edit_distance: int = 2,
        prefix_length: int = 7,
    ):
        """Initialize the spell checker.

        Args:
            min_query_length: Minimum query length to attempt correction.
            max_edit_distance: Maximum Damerau-Levenshtein distance.
            prefix_length: SymSpell prefix length for indexing.
        """
        self.min_query_length = min_query_length
        self.max_edit_distance = max_edit_distance
        self.prefix_length = prefix_length

        # Vocabulary storage
        self._vocabulary: Dict[str, VocabularyEntry] = {}
        self._normalized_to_term: Dict[str, str] = {}
        self._metaphone_index: Dict[str, Set[str]] = {}

        # SymSpell instance (lazy initialized)
        self._symspell = None

        # Common suffixes that shouldn't trigger corrections alone
        self._protected_suffixes = {
            "-seq",
            "seq",
            "-chip",
            "chip",
            "-rna",
            "rna",
            "-atac",
            "atac",
        }

    def _init_symspell(self) -> None:
        """Initialize SymSpell with vocabulary."""
        if self._symspell is not None:
            return

        symspellpy = _get_symspellpy()
        self._symspell = symspellpy.SymSpell(
            max_dictionary_edit_distance=self.max_edit_distance,
            prefix_length=self.prefix_length,
        )

        # Add all vocabulary terms
        for entry in self._vocabulary.values():
            self._symspell.create_dictionary_entry(entry.normalized, entry.frequency)

    def add_term(
        self,
        term: str,
        frequency: int = 1,
        category: str = "unknown",
    ) -> None:
        """Add a term to the vocabulary.

        Args:
            term: The term to add.
            frequency: Experiment count / popularity weight.
            category: Category label (e.g., "biosample", "target").
        """
        if not term or len(term) < 2:
            return

        normalized = term.lower().strip()

        # Skip if already exists with higher frequency
        if normalized in self._normalized_to_term:
            existing = self._vocabulary[self._normalized_to_term[normalized]]
            if existing.frequency >= frequency:
                return
            # Update existing entry
            existing.frequency = frequency
            return

        # Compute phonetic codes
        jellyfish = _get_jellyfish()
        try:
            primary, secondary = jellyfish.metaphone(normalized)
        except Exception:
            primary, secondary = "", ""

        entry = VocabularyEntry(
            term=term,
            frequency=frequency,
            category=category,
            metaphone_primary=primary or "",
            metaphone_secondary=secondary or "",
            normalized=normalized,
        )

        self._vocabulary[term] = entry
        self._normalized_to_term[normalized] = term

        # Index by phonetic codes
        if primary:
            if primary not in self._metaphone_index:
                self._metaphone_index[primary] = set()
            self._metaphone_index[primary].add(term)
        if secondary and secondary != primary:
            if secondary not in self._metaphone_index:
                self._metaphone_index[secondary] = set()
            self._metaphone_index[secondary].add(term)

        # Invalidate SymSpell cache
        self._symspell = None

    def add_terms(
        self,
        terms: List[Tuple[str, int]],
        category: str = "unknown",
    ) -> None:
        """Add multiple terms with frequencies.

        Args:
            terms: List of (term, frequency) tuples.
            category: Category for all terms.
        """
        for term, freq in terms:
            self.add_term(term, freq, category)

    def _get_phonetic_matches(self, query: str) -> Set[str]:
        """Find terms with matching phonetic codes.

        Args:
            query: The query string.

        Returns:
            Set of vocabulary terms with matching phonetic codes.
        """
        jellyfish = _get_jellyfish()
        try:
            primary, secondary = jellyfish.metaphone(query.lower())
        except Exception:
            return set()

        matches = set()
        if primary and primary in self._metaphone_index:
            matches.update(self._metaphone_index[primary])
        if secondary and secondary in self._metaphone_index:
            matches.update(self._metaphone_index[secondary])

        return matches

    def suggest(
        self,
        query: str,
        max_suggestions: int = 5,
        include_exact: bool = True,
    ) -> List[SpellingSuggestion]:
        """Get spelling suggestions for a query.

        Args:
            query: The potentially misspelled term.
            max_suggestions: Maximum number of suggestions to return.
            include_exact: Include exact matches in results.

        Returns:
            List of SpellingSuggestion objects, sorted by confidence.
        """
        if not query or len(query) < self.min_query_length:
            return []

        query_lower = query.lower().strip()

        # Check for exact match first
        if query_lower in self._normalized_to_term:
            term = self._normalized_to_term[query_lower]
            entry = self._vocabulary[term]
            if include_exact:
                return [
                    SpellingSuggestion(
                        term=entry.term,
                        distance=0,
                        confidence=1.0,
                        frequency=entry.frequency,
                        phonetic_match=True,
                        category=entry.category,
                    )
                ]
            return []

        # Skip correction for protected suffixes
        if query_lower in self._protected_suffixes:
            return []

        # Initialize SymSpell if needed
        self._init_symspell()

        suggestions: List[SpellingSuggestion] = []
        seen_terms: Set[str] = set()

        # Get phonetic matches for scoring
        phonetic_matches = self._get_phonetic_matches(query)

        # Get SymSpell suggestions
        symspellpy = _get_symspellpy()
        symspell_results = self._symspell.lookup(
            query_lower,
            symspellpy.Verbosity.CLOSEST,
            max_edit_distance=self.max_edit_distance,
        )

        for result in symspell_results:
            normalized = result.term
            if normalized not in self._normalized_to_term:
                continue

            term = self._normalized_to_term[normalized]
            if term in seen_terms:
                continue
            seen_terms.add(term)

            entry = self._vocabulary[term]
            is_phonetic = term in phonetic_matches

            # Calculate confidence score
            confidence = self._calculate_confidence(
                query_lower, entry, result.distance, is_phonetic
            )

            suggestions.append(
                SpellingSuggestion(
                    term=entry.term,
                    distance=result.distance,
                    confidence=confidence,
                    frequency=entry.frequency,
                    phonetic_match=is_phonetic,
                    category=entry.category,
                )
            )

        # Also check phonetic-only matches (may have higher edit distance)
        for term in phonetic_matches:
            if term in seen_terms:
                continue
            seen_terms.add(term)

            entry = self._vocabulary[term]
            distance = self._edit_distance(query_lower, entry.normalized)

            # Only include if within reasonable distance
            if distance <= self.max_edit_distance + 1:
                confidence = self._calculate_confidence(
                    query_lower, entry, distance, True
                )
                suggestions.append(
                    SpellingSuggestion(
                        term=entry.term,
                        distance=distance,
                        confidence=confidence,
                        frequency=entry.frequency,
                        phonetic_match=True,
                        category=entry.category,
                    )
                )

        # Sort by confidence (descending), then frequency (descending)
        suggestions.sort()

        return suggestions[:max_suggestions]

    def _calculate_confidence(
        self,
        query: str,
        entry: VocabularyEntry,
        distance: int,
        phonetic_match: bool,
    ) -> float:
        """Calculate confidence score for a suggestion.

        Args:
            query: Original query (lowercase).
            entry: Vocabulary entry.
            distance: Edit distance.
            phonetic_match: Whether phonetic codes match.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        # Base score from edit distance
        max_len = max(len(query), len(entry.normalized))
        if max_len == 0:
            return 0.0

        # Normalized distance (0 = identical, 1 = completely different)
        norm_distance = distance / max_len

        # Base confidence (inverse of normalized distance)
        confidence = 1.0 - norm_distance

        # Bonus for phonetic match (helps with scientific terms)
        if phonetic_match:
            confidence = min(1.0, confidence + 0.15)

        # Bonus for prefix match
        if entry.normalized.startswith(query[:3]) or query.startswith(
            entry.normalized[:3]
        ):
            confidence = min(1.0, confidence + 0.1)

        # Small bonus for frequency (normalized)
        # log scale to avoid dominating by very high frequencies
        import math

        freq_bonus = min(0.1, math.log10(max(1, entry.frequency)) / 50)
        confidence = min(1.0, confidence + freq_bonus)

        # Penalty for very different lengths
        len_diff = abs(len(query) - len(entry.normalized))
        if len_diff > 3:
            confidence *= 0.8

        return round(confidence, 3)

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Damerau-Levenshtein distance.

        Args:
            s1: First string.
            s2: Second string.

        Returns:
            Edit distance (insertions, deletions, substitutions, transpositions).
        """
        jellyfish = _get_jellyfish()
        return jellyfish.damerau_levenshtein_distance(s1, s2)

    def correct(self, query: str) -> str:
        """Get the best correction for a query.

        Args:
            query: The potentially misspelled term.

        Returns:
            The corrected term, or original query if no good correction found.
        """
        suggestions = self.suggest(query, max_suggestions=1, include_exact=True)
        if suggestions and suggestions[0].confidence >= 0.7:
            return suggestions[0].term
        return query

    def is_valid_term(self, term: str) -> bool:
        """Check if a term exists in the vocabulary.

        Args:
            term: Term to check.

        Returns:
            True if term exists (case-insensitive).
        """
        return term.lower().strip() in self._normalized_to_term

    @classmethod
    def from_encode_vocabularies(cls) -> "VocabularySpellChecker":
        """Create a spell checker from ENCODE vocabulary data.

        Loads terms from the ENCODE facets JSON file, including:
        - Biosamples (tissues, cell lines)
        - Targets (histone modifications, TFs)
        - Assay types
        - Organisms
        - Labs

        Returns:
            Configured VocabularySpellChecker instance.
        """
        checker = cls()

        # Import vocabulary functions
        from src.ui.vocabularies import (
            get_assay_types,
            get_biosamples,
            get_labs,
            get_life_stages,
            get_organisms,
            get_targets,
        )

        # Add biosamples (highest priority - most commonly searched)
        try:
            biosamples = get_biosamples()
            checker.add_terms(biosamples, category="biosample")
        except Exception:
            pass

        # Add targets
        try:
            targets = get_targets()
            checker.add_terms(targets, category="target")
        except Exception:
            pass

        # Add assay types
        try:
            assays = get_assay_types()
            checker.add_terms(assays, category="assay")
        except Exception:
            pass

        # Add organisms
        try:
            organisms = get_organisms()
            checker.add_terms(organisms, category="organism")
        except Exception:
            pass

        # Add life stages
        try:
            stages = get_life_stages()
            checker.add_terms(stages, category="life_stage")
        except Exception:
            pass

        # Add labs
        try:
            labs = get_labs()
            checker.add_terms(labs, category="lab")
        except Exception:
            pass

        return checker


@lru_cache(maxsize=1)
def get_spell_checker() -> VocabularySpellChecker:
    """Get or create the singleton spell checker instance.

    Returns:
        Configured VocabularySpellChecker with ENCODE vocabulary.
    """
    return VocabularySpellChecker.from_encode_vocabularies()


def suggest_correction(
    query: str, max_suggestions: int = 5
) -> List[SpellingSuggestion]:
    """Convenience function to get spelling suggestions.

    Args:
        query: The potentially misspelled term.
        max_suggestions: Maximum number of suggestions.

    Returns:
        List of SpellingSuggestion objects.
    """
    checker = get_spell_checker()
    return checker.suggest(query, max_suggestions)


def correct_spelling(query: str) -> str:
    """Convenience function to correct a spelling.

    Args:
        query: The potentially misspelled term.

    Returns:
        Corrected term or original if no good match.
    """
    checker = get_spell_checker()
    return checker.correct(query)
