# src/ui/sidebar.py
"""Sidebar UI rendering for MetaENCODE.

This module contains the sidebar filter widgets and layout,
with search execution delegated to handlers.py.
"""

import streamlit as st

from src.ui.components.initializers import get_filter_manager
from src.ui.handlers import handle_search_click
from src.ui.search_filters import FilterState
from src.ui.vocabularies import (
    HISTONE_MODIFICATIONS,
    SLIM_TYPES,
    get_assay_display_name,
    get_assay_types,
    get_biosample_names,
    get_biosamples_for_slim,
    get_lab_names,
    get_life_stages,
    get_organism_display,
    get_organisms,
    get_slim_categories,
    get_slim_display_name,
    get_targets,
)

# Module-level constants
MAX_CATEGORY_OPTIONS = 20
MAX_BIOSAMPLE_OPTIONS = 50


def render_sidebar() -> dict:
    """Render sidebar with search and filter controls.

    The sidebar includes:
    - Assay type selection with autocomplete
    - Organism selection with genome assembly labels
    - Hierarchical biosample selection (body part -> tissue)
    - Histone modification / target selection
    - Age/developmental stage search
    - More Options (lab, replicates)

    Returns:
        Dictionary containing current filter settings for backward compatibility.
    """
    filter_mgr = get_filter_manager()

    st.sidebar.title("MetaENCODE")
    st.sidebar.markdown("*ENCODE Dataset Similarity Search*")
    st.sidebar.divider()

    # --- Primary Filters (always visible) ---
    st.sidebar.subheader("Search Filters")
    st.sidebar.caption(
        "Filters apply to Search results only. "
        "Similar datasets show pure ML similarity."
    )

    # 1. Free text description search (most general filter - at top)
    description_search = st.sidebar.text_input(
        "Description search",
        value=st.session_state.filter_state.description_search or "",
        placeholder="e.g., 8-week cerebellum, H3K27ac",
        help=(
            "Search in experiment descriptions, titles, and metadata. "
            "Typos are auto-corrected."
        ),
        key="filter_description",
    )

    # Assay Type Selection - ordered by popularity (from JSON)
    assay_data = get_assay_types()  # Returns list of (name, count) tuples
    assay_options = [""] + [name for name, count in assay_data]
    assay_counts = {name: count for name, count in assay_data}
    current_assay = st.session_state.filter_state.assay_type or ""

    assay_type = st.sidebar.selectbox(
        "Assay Type",
        options=assay_options,
        index=(
            assay_options.index(current_assay) if current_assay in assay_options else 0
        ),
        format_func=lambda x: (
            "All assay types"
            if x == ""
            else f"{get_assay_display_name(x)} ({assay_counts.get(x, 0):,})"
        ),
        help="Filter by assay type (e.g., ChIP-seq, RNA-seq, Hi-C)",
        key="filter_assay_type",
    )

    # Organism Selection - Dynamic from ENCODE with experiment counts
    organism_data = get_organisms()  # Returns [(scientific_name, count), ...]
    organism_options = [""] + [name for name, _ in organism_data]
    organism_counts = {name: count for name, count in organism_data}
    current_org = st.session_state.filter_state.organism or ""

    organism = st.sidebar.selectbox(
        "Organism",
        options=organism_options,
        index=(
            organism_options.index(current_org)
            if current_org in organism_options
            else 0
        ),
        format_func=lambda x: (
            "All organisms"
            if x == ""
            else f"{get_organism_display(x)} ({organism_counts.get(x, 0):,})"
        ),
        help="Filter by organism (shows genome assembly for known organisms)",
        key="filter_organism",
    )

    # Histone Modification / Target - ordered by popularity (from JSON)
    # Get targets from JSON, but filter to show curated histone mods with descriptions
    all_targets = get_targets()  # Returns list of (name, count) tuples
    target_counts = {name: count for name, count in all_targets}

    # Show curated histone mods first (with descriptions), then other popular targets
    curated_targets = list(HISTONE_MODIFICATIONS.keys())
    other_popular_targets = [
        name for name, count in all_targets[:50] if name not in curated_targets
    ]
    target_options = [""] + curated_targets + other_popular_targets
    current_target = st.session_state.filter_state.target or ""

    def format_target(x: str) -> str:
        if x == "":
            return "All targets"
        count = target_counts.get(x, 0)
        if x in HISTONE_MODIFICATIONS:
            desc = HISTONE_MODIFICATIONS[x]["description"]
            return f"{x} - {desc} ({count:,})"
        return f"{x} ({count:,})"

    target = st.sidebar.selectbox(
        "Target / Histone Mark",
        options=target_options,
        index=(
            target_options.index(current_target)
            if current_target in target_options
            else 0
        ),
        format_func=format_target,
        help="Filter by ChIP-seq target (e.g., H3K27ac, CTCF)",
        key="filter_target",
    )

    st.sidebar.divider()

    # --- Biosample Selection (Hierarchical with Switchable Classification) ---
    st.sidebar.subheader("Biosample")

    # Classification type selector - switches between organ, cell, developmental, system
    slim_type_options = list(SLIM_TYPES.keys())
    slim_type_labels = {
        "organ": "Organ System",
        "cell": "Cell Type",
        "developmental": "Germ Layer",
        "system": "Body System",
    }

    # Initialize classification type in session state if needed
    if "classification_type" not in st.session_state:
        st.session_state.classification_type = "organ"

    classification_type = st.sidebar.selectbox(
        "Classification Type",
        options=slim_type_options,
        index=slim_type_options.index(st.session_state.classification_type),
        format_func=lambda x: slim_type_labels.get(x, x.title()),
        help=(
            "Choose how to classify biosamples "
            "(organ, cell type, germ layer, or body system)"
        ),
        key="filter_classification_type",
    )
    st.session_state.classification_type = classification_type

    # Dynamic category selector based on classification type
    category_data = get_slim_categories(classification_type)[:MAX_CATEGORY_OPTIONS]
    category_options = [""] + [name for name, _ in category_data]
    category_counts = {name: count for name, count in category_data}
    current_bp = st.session_state.filter_state.body_part or ""

    # For backwards compatibility, body_part stores the selected category
    def format_category(x: str) -> str:
        if x == "":
            return f"All {slim_type_labels[classification_type].lower()}s"
        display = get_slim_display_name(classification_type, x)
        count = category_counts.get(x, 0)
        return f"{display} ({count:,})"

    body_part = st.sidebar.selectbox(
        SLIM_TYPES[classification_type]["display_prefix"],
        options=category_options,
        index=(
            category_options.index(current_bp)
            if current_bp in category_options
            else 0
        ),
        format_func=format_category,
        help=SLIM_TYPES[classification_type]["description"],
        key="filter_body_part",
    )

    # Tissue / Cell Type (filtered by selected category)
    if body_part:
        # Get biosamples for selected category from JSON
        biosamples_data = get_biosamples_for_slim(classification_type, body_part)[
            :MAX_BIOSAMPLE_OPTIONS
        ]
        tissue_options = [""] + [name for name, _ in biosamples_data]
        biosample_counts = {name: count for name, count in biosamples_data}
    else:
        # Show top global biosamples when no category selected
        tissue_options = [""] + get_biosample_names(limit=100)
        biosample_counts = {}

    current_biosample = st.session_state.filter_state.biosample or ""

    biosample = st.sidebar.selectbox(
        "Tissue / Cell Type",
        options=tissue_options,
        index=(
            tissue_options.index(current_biosample)
            if current_biosample in tissue_options
            else 0
        ),
        format_func=lambda x: (
            "All tissues"
            if x == ""
            else (f"{x} ({biosample_counts[x]:,})" if x in biosample_counts else x)
        ),
        help="Filter by specific tissue or cell type (related terms will also match)",
        key="filter_biosample",
    )

    # Show related tissues hint if biosample is selected
    if biosample:
        related = filter_mgr.get_related_tissues(biosample)
        if len(related) > 1:
            other_related = [t for t in related if t.lower() != biosample.lower()]
            if other_related:
                st.sidebar.caption(f"Also matches: {', '.join(other_related[:3])}")

    # Life Stage - property of biosample (adult, embryonic, child, etc.)
    life_stage_data = get_life_stages()  # Returns list of (name, count) tuples
    stage_options = [""] + [name for name, count in life_stage_data]
    stage_counts = {name: count for name, count in life_stage_data}
    current_age = st.session_state.filter_state.age_stage or ""

    age_stage = st.sidebar.selectbox(
        "Life Stage",
        options=stage_options,
        index=stage_options.index(current_age) if current_age in stage_options else 0,
        format_func=lambda x: (
            "All stages" if x == "" else f"{x} ({stage_counts.get(x, 0):,})"
        ),
        help="Filter by life stage (e.g., adult, embryonic, child)",
        key="filter_age_stage",
    )

    st.sidebar.divider()

    # --- Results Control ---
    st.sidebar.subheader("Results")

    max_results = st.sidebar.slider(
        "Max results to show",
        min_value=5,
        max_value=50,
        value=st.session_state.filter_state.max_results,
        step=5,
        help="Applies to both search results and similar datasets",
        key="filter_max_results",
    )

    # --- More Options (Collapsible) ---
    with st.sidebar.expander("More Options"):
        # 7. Lab filter - ordered by popularity (from JSON)
        lab_names = get_lab_names(limit=30)  # Top 30 labs
        lab_options = [""] + lab_names
        current_lab = st.session_state.filter_state.lab or ""

        lab = st.selectbox(
            "Lab",
            options=lab_options,
            index=lab_options.index(current_lab) if current_lab in lab_options else 0,
            format_func=lambda x: "All labs" if x == "" else x,
            help="Filter by contributing lab",
            key="filter_lab",
        )

        # 8. Minimum replicates
        min_replicates = st.number_input(
            "Minimum replicates",
            min_value=0,
            max_value=10,
            value=st.session_state.filter_state.min_replicates,
            help="Filter by minimum number of replicates",
            key="filter_min_replicates",
        )

    st.sidebar.divider()

    # --- Build Filter State ---
    filter_state = FilterState(
        assay_type=assay_type if assay_type else None,
        organism=organism if organism else None,
        body_part=body_part if body_part else None,
        biosample=biosample if biosample else None,
        target=target if target else None,
        age_stage=age_stage if age_stage else None,
        lab=lab if lab else None,
        min_replicates=int(min_replicates),
        max_results=max_results,
        description_search=description_search if description_search else None,
    )

    # Store filter state
    st.session_state.filter_state = filter_state

    # Build search query preview
    search_query = filter_mgr.build_search_query(filter_state)

    if search_query:
        st.sidebar.caption(f"Search: {search_query}")

    # --- Search Button ---
    if st.sidebar.button(
        "Search ENCODE",
        type="primary",
        use_container_width=True,
        disabled=not filter_state.has_any_filter(),
    ):
        handle_search_click(filter_state, max_results)

    # Clear filters button
    if st.sidebar.button("Clear Filters", use_container_width=True):
        st.session_state.filter_state = FilterState()
        st.rerun()

    st.sidebar.divider()

    # --- Data Loading Section ---
    st.sidebar.subheader("Data")
    if st.sidebar.button("Load Sample Data", use_container_width=True):
        # Import here to avoid circular imports
        from src.utils.data_loader import load_sample_data

        load_sample_data()

    # About section
    st.sidebar.divider()
    st.sidebar.subheader("About")
    st.sidebar.markdown(
        """
        MetaENCODE uses machine learning to find similar datasets based on
        metadata embeddings. Built with SBERT and Streamlit.

        [ENCODE Portal](https://www.encodeproject.org/)
        """
    )

    # Return legacy format for backward compatibility
    return {
        "search_query": search_query,
        "organism": organism if organism else None,
        "assay_type": assay_type if assay_type else None,
        "top_n": max_results,
    }
