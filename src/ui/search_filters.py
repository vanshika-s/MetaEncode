# src/ui/search_filters.py
"""Search filter utilities for MetaENCODE.

This module provides autocomplete matching, NLP-based term matching,
and filter management for the MetaENCODE search interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

# Optional spell check integration
_spell_checker = None


def _get_spell_checker():
    """Lazy load spell checker (optional dependency)."""
    global _spell_checker
    if _spell_checker is None:
        try:
            from src.utils.spell_check import get_spell_checker

            _spell_checker = get_spell_checker()
        except ImportError:
            # Spell check dependencies not installed
            pass
    return _spell_checker


from src.ui.vocabularies import (
    ASSAY_ALIASES,
    ASSAY_TYPES,
    BODY_PARTS,
    HISTONE_ALIASES,
    HISTONE_MODIFICATIONS,
    ORGANISM_ASSEMBLIES,
    TISSUE_SYNONYMS,
    get_lab_names,
    get_life_stages,
    get_organism_display,
    get_organisms,
)


@dataclass
class FilterState:
    """Represents the current state of all search filters."""

    assay_type: Optional[str] = None
    organism: Optional[str] = None
    body_part: Optional[str] = None
    biosample: Optional[str] = None
    target: Optional[str] = None  # Histone mod or TF
    age_stage: Optional[str] = None
    lab: Optional[str] = None
    min_replicates: int = 0
    max_results: int = 20
    # Free text for description search (age, etc.)
    description_search: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for session state storage."""
        return {
            "assay_type": self.assay_type,
            "organism": self.organism,
            "body_part": self.body_part,
            "biosample": self.biosample,
            "target": self.target,
            "age_stage": self.age_stage,
            "lab": self.lab,
            "min_replicates": self.min_replicates,
            "max_results": self.max_results,
            "description_search": self.description_search,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FilterState":
        """Create from dictionary."""
        return cls(
            assay_type=d.get("assay_type"),
            organism=d.get("organism"),
            body_part=d.get("body_part"),
            biosample=d.get("biosample"),
            target=d.get("target"),
            age_stage=d.get("age_stage"),
            lab=d.get("lab"),
            min_replicates=d.get("min_replicates", 0),
            max_results=d.get("max_results", 20),
            description_search=d.get("description_search"),
        )

    def has_any_filter(self) -> bool:
        """Check if any filter is set."""
        return any(
            [
                self.assay_type,
                self.organism,
                self.body_part,
                self.biosample,
                self.target,
                self.age_stage,
                self.lab,
                self.min_replicates > 0,
                self.description_search,
            ]
        )


class SearchFilterManager:
    """Manages search filters with autocomplete and NLP matching.

    This class provides:
    - Fuzzy autocomplete matching for all filter fields
    - NLP-based synonym expansion for tissue types
    - Hierarchical biosample selection (body part -> tissue)
    - Age/developmental stage parsing and matching
    """

    def __init__(self) -> None:
        """Initialize the search filter manager."""
        self._build_search_indices()

    def _build_search_indices(self) -> None:
        """Build search indices for fast autocomplete."""
        # Build flat list of all tissues with their body parts
        self._tissue_to_body_part: Dict[str, str] = {}
        self._all_tissues: List[str] = []
        for body_part, info in BODY_PARTS.items():
            for tissue in info["tissues"]:
                self._tissue_to_body_part[tissue.lower()] = body_part
                self._all_tissues.append(tissue)

        # Build reverse synonym map
        self._synonym_map: Dict[str, Set[str]] = {}
        for term, synonyms in TISSUE_SYNONYMS.items():
            self._synonym_map[term.lower()] = {s.lower() for s in synonyms}
            # Add reverse mappings
            for syn in synonyms:
                if syn.lower() not in self._synonym_map:
                    self._synonym_map[syn.lower()] = set()
                self._synonym_map[syn.lower()].add(term.lower())

    def autocomplete_assay(self, query: str, limit: int = 10) -> List[Tuple[str, str]]:
        """Find assay types matching the query.

        Args:
            query: User input to match against.
            limit: Maximum number of results.

        Returns:
            List of (key, display_name) tuples sorted by relevance.
        """
        if not query:
            # Return common assay types
            common = [
                "ChIP-seq",
                "RNA-seq",
                "ATAC-seq",
                "Hi-C",
                "DNase-seq",
                "WGBS",
                "eCLIP",
                "CUT&RUN",
            ]
            return [(k, ASSAY_TYPES.get(k, k)) for k in common if k in ASSAY_TYPES][
                :limit
            ]

        query_lower = query.lower().strip()
        matches: List[Tuple[float, str, str]] = []

        for key, display in ASSAY_TYPES.items():
            # Check direct match
            score = self._match_score(query_lower, key.lower())

            # Check aliases
            if key in ASSAY_ALIASES:
                for alias in ASSAY_ALIASES[key]:
                    alias_score = self._match_score(query_lower, alias)
                    score = max(score, alias_score)

            # Check display name
            display_score = self._match_score(query_lower, display.lower())
            score = max(score, display_score)

            if score > 0.3:
                matches.append((score, key, display))

        matches.sort(key=lambda x: -x[0])
        return [(m[1], m[2]) for m in matches[:limit]]

    def autocomplete_organism(
        self, query: str, limit: int = 5
    ) -> List[Tuple[str, str]]:
        """Find organisms matching the query.

        Uses dynamic organism list from ENCODE API, with assembly info
        for known model organisms.

        Args:
            query: User input to match against.
            limit: Maximum number of results.

        Returns:
            List of (scientific_name, display) tuples.
        """
        # Get all organisms from ENCODE (ordered by experiment count)
        all_organisms = get_organisms()  # [(scientific_name, count), ...]

        if not query:
            # Return top organisms by experiment count
            return [
                (sci_name, get_organism_display(sci_name))
                for sci_name, _ in all_organisms
            ][:limit]

        query_lower = query.lower().strip()
        matches: List[Tuple[float, str, str]] = []

        for sci_name, count in all_organisms:
            # Get common name if available
            assembly_info = ORGANISM_ASSEMBLIES.get(sci_name, {})
            common_name = assembly_info.get("common_name", "")
            short_name = assembly_info.get("short_name", "")
            assembly = assembly_info.get("assembly", "")

            # Score based on matching scientific name, common name, or assembly
            score = max(
                self._match_score(query_lower, sci_name.lower()),
                (
                    self._match_score(query_lower, common_name.lower())
                    if common_name
                    else 0
                ),
                self._match_score(query_lower, short_name.lower()) if short_name else 0,
                self._match_score(query_lower, assembly.lower()) if assembly else 0,
            )
            if score > 0.3:
                display = get_organism_display(sci_name)
                matches.append((score, sci_name, display))

        matches.sort(key=lambda x: -x[0])
        return [(m[1], m[2]) for m in matches[:limit]]

    def autocomplete_target(self, query: str, limit: int = 15) -> List[Tuple[str, str]]:
        """Find histone modifications or TFs matching the query.

        Args:
            query: User input (e.g., "H3K27", "CTCF").
            limit: Maximum number of results.

        Returns:
            List of (key, description) tuples.
        """
        if not query:
            # Return common targets
            common = [
                "H3K27ac",
                "H3K4me3",
                "H3K4me1",
                "H3K27me3",
                "H3K9me3",
                "H3K36me3",
                "CTCF",
                "POLR2A",
            ]
            return [
                (k, HISTONE_MODIFICATIONS[k]["description"])
                for k in common
                if k in HISTONE_MODIFICATIONS
            ][:limit]

        query_lower = query.lower().strip()
        matches: List[Tuple[float, str, str]] = []

        for key, info in HISTONE_MODIFICATIONS.items():
            score = max(
                self._match_score(query_lower, key.lower()),
                self._match_score(query_lower, info["full_name"].lower()),
            )

            # Check aliases
            if key in HISTONE_ALIASES:
                for alias in HISTONE_ALIASES[key]:
                    alias_score = self._match_score(query_lower, alias)
                    score = max(score, alias_score)

            if score > 0.3:
                matches.append((score, key, info["description"]))

        matches.sort(key=lambda x: -x[0])
        return [(m[1], m[2]) for m in matches[:limit]]

    def autocomplete_body_part(
        self, query: str, limit: int = 10
    ) -> List[Tuple[str, str]]:
        """Find body parts/organ systems matching the query.

        Args:
            query: User input (e.g., "brain", "blood").
            limit: Maximum number of results.

        Returns:
            List of (key, display_name) tuples.
        """
        if not query:
            return [(k, v["display_name"]) for k, v in BODY_PARTS.items()][:limit]

        query_lower = query.lower().strip()
        matches: List[Tuple[float, str, str]] = []

        for key, info in BODY_PARTS.items():
            score = max(
                self._match_score(query_lower, key),
                self._match_score(query_lower, info["display_name"].lower()),
            )

            # Check aliases
            for alias in info.get("aliases", []):
                alias_score = self._match_score(query_lower, alias)
                score = max(score, alias_score)

            # Check if query matches any tissues in this body part
            for tissue in info["tissues"]:
                tissue_score = self._match_score(query_lower, tissue.lower())
                if tissue_score > 0.7:
                    score = max(score, tissue_score * 0.9)

            if score > 0.3:
                matches.append((score, key, info["display_name"]))

        matches.sort(key=lambda x: -x[0])
        return [(m[1], m[2]) for m in matches[:limit]]

    def autocomplete_biosample(
        self, query: str, body_part: Optional[str] = None, limit: int = 15
    ) -> List[Tuple[str, str]]:
        """Find biosamples/tissues matching the query.

        If body_part is specified, only returns tissues from that body part.
        Otherwise searches all tissues and includes related synonyms.

        Args:
            query: User input (e.g., "cerebellum", "K562").
            body_part: Optional body part to restrict search.
            limit: Maximum number of results.

        Returns:
            List of (tissue_name, body_part_name) tuples.
        """
        # Get tissues to search
        if body_part and body_part in BODY_PARTS:
            tissues = BODY_PARTS[body_part]["tissues"]
            bp_name = BODY_PARTS[body_part]["display_name"]
        else:
            tissues = self._all_tissues
            bp_name = None

        if not query:
            return [(t, bp_name or self._get_body_part_display(t)) for t in tissues][
                :limit
            ]

        query_lower = query.lower().strip()
        matches: List[Tuple[float, str, str]] = []
        seen = set()

        for tissue in tissues:
            tissue_lower = tissue.lower()
            if tissue_lower in seen:
                continue

            score = self._match_score(query_lower, tissue_lower)

            # Check synonyms for this tissue
            if tissue_lower in self._synonym_map:
                for syn in self._synonym_map[tissue_lower]:
                    syn_score = self._match_score(query_lower, syn)
                    score = max(score, syn_score * 0.95)

            if score > 0.3:
                display_bp = bp_name or self._get_body_part_display(tissue)
                matches.append((score, tissue, display_bp))
                seen.add(tissue_lower)

        matches.sort(key=lambda x: -x[0])
        return [(m[1], m[2]) for m in matches[:limit]]

    def get_related_tissues(self, tissue: str) -> List[str]:
        """Get tissues related by synonym to the given tissue.

        This enables NLP-like matching where "cerebellum" also matches
        "hindbrain" and vice versa.

        Args:
            tissue: Tissue name to find relatives for.

        Returns:
            List of related tissue names including the original.
        """
        tissue_lower = tissue.lower()
        related = {tissue}

        if tissue_lower in self._synonym_map:
            for syn in self._synonym_map[tissue_lower]:
                # Find the canonical form
                for t in self._all_tissues:
                    if t.lower() == syn:
                        related.add(t)
                        break

        return list(related)

    def autocomplete_age(
        self, query: str, organism: Optional[str] = None, limit: int = 15
    ) -> List[Tuple[str, str]]:
        """Find life stages matching the query.

        Note: Returns actual ENCODE life stages (adult, embryonic, child, etc.),
        not fabricated developmental stages like E14.5 or P56.

        Args:
            query: User input (e.g., "adult", "embryonic", "child").
            organism: Optional organism to filter stages (currently unused).
            limit: Maximum number of results.

        Returns:
            List of (stage_name, count_info) tuples.
        """
        # Get life stages from JSON (ordered by experiment count)
        life_stages = get_life_stages()

        if not query:
            # Return stages ordered by popularity
            return [(name, f"{count:,} experiments") for name, count in life_stages][
                :limit
            ]

        query_lower = query.lower().strip()
        matches: List[Tuple[float, str, str]] = []

        for name, count in life_stages:
            score = self._match_score(query_lower, name.lower())
            if score > 0.3:
                matches.append((score, name, f"{count:,} experiments"))

        matches.sort(key=lambda x: -x[0])
        return [(m[1], m[2]) for m in matches[:limit]]

    def autocomplete_lab(self, query: str, limit: int = 10) -> List[str]:
        """Find labs matching the query.

        Args:
            query: User input.
            limit: Maximum number of results.

        Returns:
            List of lab names (ordered by experiment count).
        """
        # Get labs from JSON (ordered by experiment count)
        labs = get_lab_names(limit=50)  # Get more than limit for filtering

        if not query:
            return labs[:limit]

        query_lower = query.lower().strip()
        matches: List[Tuple[float, str]] = []

        for lab in labs:
            score = self._match_score(query_lower, lab.lower())
            if score > 0.2:
                matches.append((score, lab))

        matches.sort(key=lambda x: -x[0])
        return [m[1] for m in matches[:limit]]

    def _match_score(self, query: str, target: str) -> float:
        """Calculate fuzzy match score between query and target.

        Uses a combination of:
        - Prefix matching (high score for matching start)
        - Substring matching (medium score for containing)
        - SequenceMatcher ratio (lower score for partial matches)

        Args:
            query: Search query (lowercase).
            target: Target string to match (lowercase).

        Returns:
            Score between 0 and 1, higher is better match.
        """
        if not query or not target:
            return 0.0

        # Exact match
        if query == target:
            return 1.0

        # Prefix match (query starts target or target starts with query)
        if target.startswith(query):
            return 0.9 + 0.1 * (len(query) / len(target))

        if query.startswith(target):
            return 0.8

        # Contains match
        if query in target:
            return 0.7 + 0.1 * (len(query) / len(target))

        if target in query:
            return 0.6

        # Word boundary match
        # Note: This branch is logically unreachable because if word.startswith(query),
        # then query is in target (at the start of a word), which is caught by line 492.
        # Kept for defensive programming in case the above logic changes.
        target_words = target.split()
        for word in target_words:
            if word.startswith(query):
                return 0.65  # pragma: no cover

        # Fuzzy match using SequenceMatcher
        ratio = SequenceMatcher(None, query, target).ratio()
        return ratio * 0.5

    def _fuzzy_text_search(
        self, df: pd.DataFrame, search_terms: List[str], threshold: float = 0.75
    ) -> pd.Series:
        """Search text columns with fuzzy matching and spell correction fallback.

        Args:
            df: DataFrame to search.
            search_terms: List of search terms (already lowercased).
            threshold: Minimum SequenceMatcher ratio for fuzzy matching (0.0-1.0).
                       Default 0.75 balances catching typos vs false positives.

        Returns:
            Boolean mask of matching rows.
        """
        # Columns to search (in priority order)
        search_cols = [
            "description",
            "title",
            "combined_text",
            "assay_term_name",
            "biosample_term_name",
            "organism",
            "lab",
        ]
        available_cols = [c for c in search_cols if c in df.columns]

        if not available_cols:
            return pd.Series(True, index=df.index)

        # Build combined search text per row
        def combine_text(row: pd.Series) -> str:
            return " ".join(str(row.get(c, "") or "") for c in available_cols).lower()

        combined = df.apply(combine_text, axis=1)

        # Try to get spell-corrected terms
        spell_checker = _get_spell_checker()
        corrected_terms: Dict[str, str] = {}
        if spell_checker:
            for term in search_terms:
                corrected = spell_checker.correct(term)
                if corrected != term:
                    corrected_terms[term] = corrected.lower()

        # For each term, find matches
        mask = pd.Series(True, index=df.index)
        for term in search_terms:
            # Fast path: exact substring match
            term_mask = combined.str.contains(term, case=False, na=False, regex=False)

            # Also check spell-corrected term if available
            if term in corrected_terms:
                corrected = corrected_terms[term]
                term_mask = term_mask | combined.str.contains(
                    corrected, case=False, na=False, regex=False
                )

            # Fuzzy fallback for non-matches
            if not term_mask.all():
                for idx in df.index[~term_mask]:
                    text = combined.loc[idx]
                    # Check each word for fuzzy match
                    for word in text.split():
                        # Use SequenceMatcher directly (not _match_score)
                        ratio = SequenceMatcher(None, term, word).ratio()
                        if ratio >= threshold:
                            term_mask.loc[idx] = True
                            break
                        # Also check corrected term if available
                        if term in corrected_terms:
                            ratio = SequenceMatcher(
                                None, corrected_terms[term], word
                            ).ratio()
                            if ratio >= threshold:
                                term_mask.loc[idx] = True
                                break

            mask = mask & term_mask

        return mask

    def _get_body_part_display(self, tissue: str) -> str:
        """Get the display name of the body part containing a tissue."""
        tissue_lower = tissue.lower()
        if tissue_lower in self._tissue_to_body_part:
            bp_key = self._tissue_to_body_part[tissue_lower]
            display: str = BODY_PARTS[bp_key]["display_name"]
            return display
        return ""

    def apply_filters(
        self,
        df: pd.DataFrame,
        filters: FilterState,
        search_mode: bool = False,
    ) -> pd.DataFrame:
        """Apply filter state to a DataFrame.

        This method handles filtering for both search results and
        similarity results, with smart matching for biosample synonyms.

        Args:
            df: DataFrame to filter.
            filters: Current filter state.
            search_mode: If True, apply more lenient matching for search.

        Returns:
            Filtered DataFrame.
        """
        if df.empty:
            return df

        result = df.copy()

        # Organism filter
        if filters.organism and "organism" in result.columns:
            result = result[result["organism"].str.lower() == filters.organism.lower()]

        # Assay type filter
        if filters.assay_type and "assay_term_name" in result.columns:
            # Handle Hi-C / HiC variants
            assay = filters.assay_type
            if assay.lower() in ("hi-c", "hic"):
                result = result[
                    result["assay_term_name"]
                    .str.lower()
                    .isin(["hi-c", "hic", "in situ hi-c"])
                ]
            else:
                result = result[result["assay_term_name"].str.lower() == assay.lower()]

        # Biosample filter with synonym expansion
        if filters.biosample and "biosample_term_name" in result.columns:
            # Get related tissues
            related = self.get_related_tissues(filters.biosample)
            related_lower = {t.lower() for t in related}

            result = result[
                result["biosample_term_name"].str.lower().isin(related_lower)
            ]

        # Body part / organ filter (filters to all biosamples under that organ)
        if filters.body_part and "biosample_term_name" in result.columns:
            from src.ui.vocabularies import get_biosamples_for_organ

            organ_biosamples = get_biosamples_for_organ(filters.body_part)
            valid_biosamples = {name.lower() for name, _ in organ_biosamples}
            if valid_biosamples:
                result = result[
                    result["biosample_term_name"].str.lower().isin(valid_biosamples)
                ]

        # Target (histone mod) filter - search in description
        if filters.target:
            target = filters.target
            # Search in description and title
            mask = pd.Series(False, index=result.index)
            for col in ["description", "title", "combined_text"]:
                if col in result.columns:
                    mask = mask | result[col].str.contains(target, case=False, na=False)
            result = result[mask]

        # Life stage filter - use the life_stage column directly
        if filters.age_stage and "life_stage" in result.columns:
            result = result[
                result["life_stage"].str.lower() == filters.age_stage.lower()
            ]

        # Description search with fuzzy matching
        if filters.description_search:
            search_terms = filters.description_search.lower().split()
            mask = self._fuzzy_text_search(result, search_terms, threshold=0.75)
            result = result[mask]

        # Lab filter
        if filters.lab and "lab" in result.columns:
            lab_lower = filters.lab.lower()
            result = result[result["lab"].str.lower().str.contains(lab_lower, na=False)]

        # Replicate count filter
        if filters.min_replicates > 0 and "replicate_count" in result.columns:
            result = result[result["replicate_count"] >= filters.min_replicates]

        return result

    def build_search_query(self, filters: FilterState) -> str:
        """Build a human-readable search summary from filter state.

        This is used for display purposes only (showing the user what filters
        are active). The actual API call uses structured parameters via
        fetch_experiments(), not free-text search.

        Args:
            filters: Current filter state.

        Returns:
            Human-readable summary of active filters.
        """
        parts = []

        if filters.assay_type:
            parts.append(filters.assay_type)

        if filters.organism:
            parts.append(filters.organism)

        if filters.target:
            parts.append(filters.target)

        if filters.biosample:
            parts.append(filters.biosample)

        if filters.age_stage:
            parts.append(f"age:{filters.age_stage}")

        if filters.description_search:
            parts.append(f'"{filters.description_search}"')

        return " ".join(parts) if parts else ""


def parse_age_from_text(text: str) -> Optional[str]:
    """Extract age/developmental stage from free text.

    Parses descriptions like "8-week mouse cerebellum" to extract
    age information.

    Args:
        text: Text to parse.

    Returns:
        Extracted age string or None if not found.
    """
    if not text:
        return None

    text_lower = text.lower()

    # Common age patterns
    patterns = [
        # Mouse postnatal (P0, P7, P56, etc.)
        r"\b(p\d+(?:\.\d+)?)\b",
        # Mouse embryonic (E10.5, E14.5, etc.)
        r"\b(e\d+(?:\.\d+)?)\b",
        # Week-based (8 week, 8-week, 8weeks)
        r"\b(\d+)[\s-]?weeks?\b",
        # Month-based (2 month, 3-month)
        r"\b(\d+)[\s-]?months?\b",
        # Age terms
        r"\b(adult|newborn|embryonic|fetal|juvenile|aged)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            result = match.group(0)
            # Normalize format
            if re.match(r"^\d+[\s-]?weeks?$", result):
                num = re.match(r"(\d+)", result).group(1)
                return f"{num} weeks"
            if re.match(r"^\d+[\s-]?months?$", result):
                num = re.match(r"(\d+)", result).group(1)
                return f"{num} months"
            # Only uppercase P-day/E-day patterns (e.g., p0, e14.5), not words
            if re.match(r"^[ep]\d", result):
                return result.upper()
            return result

    return None
