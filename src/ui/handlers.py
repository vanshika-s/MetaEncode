# src/ui/handlers.py
"""Search execution and API call handlers for MetaENCODE.

This module contains the business logic for search operations,
separated from the UI rendering in sidebar.py.
"""

import pandas as pd
import streamlit as st

from src.ui.components.initializers import get_api_client, get_filter_manager
from src.ui.search_filters import FilterState
from src.utils.spell_check import correct_spelling


def apply_spell_correction(search_term: str) -> tuple[str, str | None]:
    """Apply spell correction to search terms.

    Args:
        search_term: The original search term to correct.

    Returns:
        Tuple of (corrected_search, correction_message).
        correction_message is None if no corrections were made.
    """
    if not search_term:
        return "", None

    words = search_term.split()
    corrected_words = []
    corrections_made = []

    for word in words:
        corrected = correct_spelling(word)
        corrected_words.append(corrected)
        if corrected.lower() != word.lower():
            corrections_made.append(f"{word} -> {corrected}")

    corrected_search = " ".join(corrected_words)
    correction_msg = None
    if corrections_made:
        correction_msg = f"Corrected: {', '.join(corrections_made)}"

    return corrected_search, correction_msg


def execute_search(
    filter_state: FilterState,
    max_results: int,
) -> tuple[pd.DataFrame, str | None]:
    """Execute ENCODE search with the given filters.

    Args:
        filter_state: Current filter state with all search parameters.
        max_results: Maximum number of results to return.

    Returns:
        Tuple of (results_df, spell_correction_message).

    Raises:
        Exception: If the search fails.
    """
    client = get_api_client()
    filter_mgr = get_filter_manager()

    # Map organism key to scientific name for API
    organism_scientific = None
    if filter_state.organism:
        org_map = {
            "human": "Homo sapiens",
            "mouse": "Mus musculus",
            "fly": "Drosophila melanogaster",
            "worm": "Caenorhabditis elegans",
        }
        organism_scientific = org_map.get(filter_state.organism, filter_state.organism)

    # Apply spell correction to search terms
    corrected_search = None
    spell_correction_msg = None
    if filter_state.description_search:
        corrected_search, spell_correction_msg = apply_spell_correction(
            filter_state.description_search
        )

    # Use fetch_experiments with all available API parameters
    results = client.fetch_experiments(
        assay_type=filter_state.assay_type,
        organism=organism_scientific,
        biosample=filter_state.biosample,
        target=filter_state.target,
        life_stage=filter_state.age_stage,
        search_term=corrected_search,
        limit=max(max_results * 5, 200),  # Fetch more for filtering
    )

    # Apply post-filtering for body_part, target, etc.
    # Note: age_stage is NOT included here because API already filtered by it
    if not results.empty:
        # Apply non-API filters (body_part, description, lab, replicates)
        post_filter_state = FilterState(
            body_part=filter_state.body_part,
            target=filter_state.target,
            # age_stage excluded - API already filtered by life_stage
            lab=filter_state.lab,
            min_replicates=filter_state.min_replicates,
            description_search=filter_state.description_search,
        )
        if post_filter_state.has_any_filter():
            results = filter_mgr.apply_filters(
                results, post_filter_state, search_mode=True
            )
        results = results.head(max_results)

    return results, spell_correction_msg


def handle_search_click(filter_state: FilterState, max_results: int) -> None:
    """Handle search button click event.

    Updates session state with search results and displays status messages.

    Args:
        filter_state: Current filter state.
        max_results: Maximum results to return.
    """
    if not filter_state.has_any_filter():
        st.sidebar.warning("Please set at least one filter")
        return

    with st.spinner("Searching ENCODE..."):
        try:
            results, spell_msg = execute_search(filter_state, max_results)
            st.session_state.search_results = results
            if spell_msg:
                st.sidebar.info(spell_msg)
            st.sidebar.success(f"Found {len(results)} results")
        except Exception as e:
            st.sidebar.error(f"Search failed: {e}")
