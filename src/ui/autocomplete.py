# src/ui/autocomplete.py
"""Autocomplete provider for MetaENCODE search interface.

This module provides VS Code-like autocomplete functionality by combining:
- Spell checking for typo correction
- Fuzzy matching for partial queries
- Category-aware suggestions (biosample, target, assay, etc.)
- Popularity ranking based on experiment counts

Example usage with streamlit-searchbox:
    from streamlit_searchbox import st_searchbox
    from src.ui.autocomplete import AutocompleteProvider

    provider = AutocompleteProvider()

    def search_fn(query: str) -> List[Tuple[str, str]]:
        suggestions = provider.get_suggestions(query, field="biosample")
        return [(s["value"], s["display"]) for s in suggestions]

    selected = st_searchbox(search_fn, key="biosample_search")
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from src.ui.search_filters import SearchFilterManager
from src.ui.vocabularies import (
    ASSAY_ALIASES,
    BODY_PARTS,
    HISTONE_ALIASES,
    ORGANISM_ASSEMBLIES,
    get_assay_types,
    get_biosamples,
    get_biosamples_for_organ,
    get_labs,
    get_life_stages,
    get_organ_display_name,
    get_organ_systems,
    get_organisms,
    get_targets,
)


@dataclass
class AutocompleteSuggestion:
    """A suggestion for autocomplete display."""

    value: str  # The value to use when selected
    display: str  # Display text in dropdown
    category: Optional[str] = None  # Category label (e.g., "Brain")
    count: Optional[int] = None  # Experiment count
    is_correction: bool = False  # True if this is a spelling correction
    match_type: str = "exact"  # "exact", "prefix", "fuzzy", "spelling"
    confidence: float = 1.0  # Match confidence (0.0 to 1.0)


class AutocompleteProvider:
    """Provides autocomplete suggestions for search fields.

    Combines multiple matching strategies:
    1. Exact prefix matching (highest priority)
    2. Fuzzy substring matching
    3. Spell correction for typos
    4. Alias/synonym matching

    Results are ranked by:
    - Match type (exact > prefix > fuzzy > spelling)
    - Experiment count (popularity)
    - Confidence score
    """

    def __init__(self, use_spell_check: bool = True):
        """Initialize the autocomplete provider.

        Args:
            use_spell_check: Whether to enable spell checking suggestions.
        """
        self._filter_manager = SearchFilterManager()
        self._use_spell_check = use_spell_check
        self._spell_checker = None

        # Cache for vocabulary data
        self._vocab_cache: Dict[str, List[Tuple[str, int]]] = {}

    def _get_spell_checker(self):
        """Lazy load spell checker."""
        if self._spell_checker is None and self._use_spell_check:
            try:
                from src.utils.spell_check import get_spell_checker

                self._spell_checker = get_spell_checker()
            except ImportError:
                self._use_spell_check = False
        return self._spell_checker

    def _get_vocabulary(self, field: str) -> List[Tuple[str, int]]:
        """Get vocabulary for a field.

        Args:
            field: Field name (biosample, target, assay, organism, lab, life_stage, organ)

        Returns:
            List of (term, count) tuples.
        """
        if field in self._vocab_cache:
            return self._vocab_cache[field]

        vocab: List[Tuple[str, int]] = []

        if field == "biosample":
            vocab = get_biosamples()
        elif field == "target":
            vocab = get_targets()
        elif field == "assay":
            vocab = get_assay_types()
        elif field == "organism":
            vocab = get_organisms()
        elif field == "lab":
            vocab = get_labs()
        elif field == "life_stage":
            vocab = get_life_stages()
        elif field == "organ":
            vocab = get_organ_systems()

        self._vocab_cache[field] = vocab
        return vocab

    def get_suggestions(
        self,
        query: str,
        field: str,
        limit: int = 10,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Get autocomplete suggestions for a query.

        Args:
            query: User input query.
            field: Field to search (biosample, target, assay, organism, lab, life_stage, organ).
            limit: Maximum number of suggestions.
            context: Optional context (e.g., {"organ": "brain"} for biosample field).

        Returns:
            List of suggestion dictionaries with keys:
            - value: The value to use when selected
            - display: Display text
            - category: Optional category label
            - count: Experiment count
            - is_correction: Whether this is a spelling correction
            - match_type: Type of match (exact, prefix, fuzzy, spelling)
        """
        if not query:
            return self._get_default_suggestions(field, limit, context)

        query_lower = query.lower().strip()
        suggestions: List[AutocompleteSuggestion] = []
        seen_values: set = set()

        # Get vocabulary for this field
        vocab = self._get_vocabulary(field)
        if context and field == "biosample" and "organ" in context:
            # Filter biosamples by organ
            organ = context["organ"]
            organ_biosamples = get_biosamples_for_organ(organ)
            if organ_biosamples:
                vocab = organ_biosamples

        # 1. Exact prefix matches (highest priority)
        for term, count in vocab:
            term_lower = term.lower()
            if term_lower.startswith(query_lower):
                if term not in seen_values:
                    seen_values.add(term)
                    suggestions.append(
                        AutocompleteSuggestion(
                            value=term,
                            display=self._format_display(term, count, field),
                            category=self._get_category(term, field),
                            count=count,
                            is_correction=False,
                            match_type="prefix",
                            confidence=1.0,
                        )
                    )

        # 2. Contains matches
        for term, count in vocab:
            if term in seen_values:
                continue
            term_lower = term.lower()
            if query_lower in term_lower:
                seen_values.add(term)
                # Score based on position of match
                pos = term_lower.find(query_lower)
                confidence = 0.9 - (pos / len(term_lower)) * 0.2
                suggestions.append(
                    AutocompleteSuggestion(
                        value=term,
                        display=self._format_display(term, count, field),
                        category=self._get_category(term, field),
                        count=count,
                        is_correction=False,
                        match_type="fuzzy",
                        confidence=confidence,
                    )
                )

        # 3. Alias matching
        alias_matches = self._match_aliases(query_lower, field)
        for term, confidence in alias_matches:
            if term in seen_values:
                continue
            seen_values.add(term)
            # Find count for this term
            count = next((c for t, c in vocab if t == term), 0)
            suggestions.append(
                AutocompleteSuggestion(
                    value=term,
                    display=self._format_display(term, count, field),
                    category=self._get_category(term, field),
                    count=count,
                    is_correction=False,
                    match_type="fuzzy",
                    confidence=confidence,
                )
            )

        # 4. Spell correction (if enabled and few matches so far)
        if self._use_spell_check and len(suggestions) < limit // 2:
            spell_checker = self._get_spell_checker()
            if spell_checker:
                spell_suggestions = spell_checker.suggest(query, max_suggestions=5)
                for ss in spell_suggestions:
                    if ss.term in seen_values:
                        continue
                    # Only include if it's a real correction (distance > 0)
                    if ss.distance > 0 and ss.confidence >= 0.6:
                        seen_values.add(ss.term)
                        # Find count for this term
                        count = next(
                            (c for t, c in vocab if t == ss.term), ss.frequency
                        )
                        suggestions.append(
                            AutocompleteSuggestion(
                                value=ss.term,
                                display=self._format_correction_display(
                                    ss.term, query, count, field
                                ),
                                category=self._get_category(ss.term, field),
                                count=count,
                                is_correction=True,
                                match_type="spelling",
                                confidence=ss.confidence,
                            )
                        )

        # Sort suggestions
        suggestions.sort(
            key=lambda s: (
                -self._match_type_priority(s.match_type),
                -s.confidence,
                -(s.count or 0),
            )
        )

        # Convert to dictionaries
        return [
            {
                "value": s.value,
                "display": s.display,
                "category": s.category,
                "count": s.count,
                "is_correction": s.is_correction,
                "match_type": s.match_type,
            }
            for s in suggestions[:limit]
        ]

    def _match_type_priority(self, match_type: str) -> int:
        """Get priority for a match type (higher = better)."""
        priorities = {
            "exact": 4,
            "prefix": 3,
            "fuzzy": 2,
            "spelling": 1,
        }
        return priorities.get(match_type, 0)

    def _get_default_suggestions(
        self,
        field: str,
        limit: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Get default suggestions when query is empty.

        Returns top items by experiment count.
        """
        vocab = self._get_vocabulary(field)
        if context and field == "biosample" and "organ" in context:
            organ = context["organ"]
            organ_biosamples = get_biosamples_for_organ(organ)
            if organ_biosamples:
                vocab = organ_biosamples

        suggestions = []
        for term, count in vocab[:limit]:
            suggestions.append(
                {
                    "value": term,
                    "display": self._format_display(term, count, field),
                    "category": self._get_category(term, field),
                    "count": count,
                    "is_correction": False,
                    "match_type": "exact",
                }
            )
        return suggestions

    def _format_display(self, term: str, count: int, field: str) -> str:
        """Format term for display in dropdown."""
        if count and count > 0:
            return f"{term} ({count:,})"
        return term

    def _format_correction_display(
        self, term: str, original: str, count: int, field: str
    ) -> str:
        """Format spelling correction for display."""
        if count and count > 0:
            return f"{term} ({count:,})"
        return f"{term}"

    def _get_category(self, term: str, field: str) -> Optional[str]:
        """Get category label for a term."""
        if field == "biosample":
            # Try to find organ for biosample
            from src.ui.vocabularies import get_primary_organ_for_biosample

            organ = get_primary_organ_for_biosample(term)
            if organ:
                return get_organ_display_name(organ)
        elif field == "organism":
            info = ORGANISM_ASSEMBLIES.get(term, {})
            if info.get("assembly"):
                return info["assembly"]
        elif field == "organ":
            return get_organ_display_name(term)

        return None

    def _match_aliases(self, query: str, field: str) -> List[Tuple[str, float]]:
        """Match query against aliases.

        Returns:
            List of (canonical_term, confidence) tuples.
        """
        matches: List[Tuple[str, float]] = []

        if field == "assay":
            for canonical, aliases in ASSAY_ALIASES.items():
                for alias in aliases:
                    if query in alias.lower() or alias.lower().startswith(query):
                        confidence = 0.85 if alias.lower().startswith(query) else 0.7
                        matches.append((canonical, confidence))
                        break

        elif field == "target":
            for canonical, aliases in HISTONE_ALIASES.items():
                for alias in aliases:
                    if query in alias.lower() or alias.lower().startswith(query):
                        confidence = 0.85 if alias.lower().startswith(query) else 0.7
                        matches.append((canonical, confidence))
                        break

        elif field == "organism":
            # Match by common name or assembly
            for sci_name, info in ORGANISM_ASSEMBLIES.items():
                common = info.get("common_name", "").lower()
                assembly = info.get("assembly", "").lower()
                if query in common or query in assembly:
                    confidence = 0.9 if common.startswith(query) else 0.8
                    matches.append((sci_name, confidence))

        return matches


@lru_cache(maxsize=1)
def get_autocomplete_provider() -> AutocompleteProvider:
    """Get or create the singleton autocomplete provider.

    Returns:
        Configured AutocompleteProvider instance.
    """
    return AutocompleteProvider()


# Convenience functions for streamlit-searchbox integration


def create_biosample_search_fn(
    organ: Optional[str] = None,
) -> callable:
    """Create a search function for biosample autocomplete.

    Args:
        organ: Optional organ to filter biosamples.

    Returns:
        Search function compatible with streamlit-searchbox.
    """
    provider = get_autocomplete_provider()
    context = {"organ": organ} if organ else None

    def search_fn(query: str) -> List[Tuple[str, str]]:
        suggestions = provider.get_suggestions(
            query, field="biosample", limit=15, context=context
        )
        return [(s["value"], s["display"]) for s in suggestions]

    return search_fn


def create_target_search_fn() -> callable:
    """Create a search function for target/histone autocomplete.

    Returns:
        Search function compatible with streamlit-searchbox.
    """
    provider = get_autocomplete_provider()

    def search_fn(query: str) -> List[Tuple[str, str]]:
        suggestions = provider.get_suggestions(query, field="target", limit=15)
        return [(s["value"], s["display"]) for s in suggestions]

    return search_fn


def create_assay_search_fn() -> callable:
    """Create a search function for assay type autocomplete.

    Returns:
        Search function compatible with streamlit-searchbox.
    """
    provider = get_autocomplete_provider()

    def search_fn(query: str) -> List[Tuple[str, str]]:
        suggestions = provider.get_suggestions(query, field="assay", limit=15)
        return [(s["value"], s["display"]) for s in suggestions]

    return search_fn


def create_organism_search_fn() -> callable:
    """Create a search function for organism autocomplete.

    Returns:
        Search function compatible with streamlit-searchbox.
    """
    provider = get_autocomplete_provider()

    def search_fn(query: str) -> List[Tuple[str, str]]:
        suggestions = provider.get_suggestions(query, field="organism", limit=10)
        return [(s["value"], s["display"]) for s in suggestions]

    return search_fn


def create_lab_search_fn() -> callable:
    """Create a search function for lab autocomplete.

    Returns:
        Search function compatible with streamlit-searchbox.
    """
    provider = get_autocomplete_provider()

    def search_fn(query: str) -> List[Tuple[str, str]]:
        suggestions = provider.get_suggestions(query, field="lab", limit=15)
        return [(s["value"], s["display"]) for s in suggestions]

    return search_fn


def create_organ_search_fn() -> callable:
    """Create a search function for organ/body part autocomplete.

    Returns:
        Search function compatible with streamlit-searchbox.
    """
    provider = get_autocomplete_provider()

    def search_fn(query: str) -> List[Tuple[str, str]]:
        suggestions = provider.get_suggestions(query, field="organ", limit=15)
        return [(s["value"], s["display"]) for s in suggestions]

    return search_fn
