# src/ui/__init__.py
"""UI components and utilities for MetaENCODE.

Note: Some modules are NOT imported at the top level to avoid circular imports.
Import directly when needed:
    - from src.ui.components.initializers import get_cache_manager, ...
    - from src.ui.components.session import init_session_state
    - from src.ui.sidebar import render_sidebar
    - from src.ui.handlers import execute_search, ...
    - from src.ui.tabs import render_search_tab, render_similar_tab, render_visualize_tab
    - from src.ui.formatters import format_organism_display, truncate_text
"""

# Autocomplete provider (uses spell check if available)
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
from src.ui.search_filters import FilterState, SearchFilterManager
from src.ui.vocabularies import (
    ASSAY_TYPES,
    BODY_PARTS,
    HISTONE_MODIFICATIONS,
    LIFE_STAGES,
    ORGANISM_ASSEMBLIES,
    ORGANISMS,
    TISSUE_SYNONYMS,
    get_assay_types,
    get_life_stages,
    get_organism_display,
    get_organisms,
)

__all__ = [
    # Search filters
    "FilterState",
    "SearchFilterManager",
    # Vocabularies
    "ASSAY_TYPES",
    "ORGANISM_ASSEMBLIES",
    "ORGANISMS",
    "HISTONE_MODIFICATIONS",
    "BODY_PARTS",
    "TISSUE_SYNONYMS",
    "LIFE_STAGES",
    "get_assay_types",
    "get_life_stages",
    "get_organism_display",
    "get_organisms",
]
