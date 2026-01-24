# app.py
"""MetaENCODE: ENCODE Dataset Similarity Search Application.

This Streamlit application enables researchers to discover related ENCODE
datasets through metadata-driven similarity scoring. Users can search for
datasets, select a seed dataset, and explore similar experiments through
interactive visualizations.

Run with: streamlit run app.py
"""

import pandas as pd
import streamlit as st

from src.api.encode_client import EncodeClient
from src.ml.embeddings import EmbeddingGenerator
from src.ml.feature_combiner import FeatureCombiner
from src.ml.similarity import SimilarityEngine
from src.processing.metadata import MetadataProcessor
from src.ui.search_filters import FilterState, SearchFilterManager
from src.ui.vocabularies import (
    BODY_PARTS,
    HISTONE_MODIFICATIONS,
    ORGANISMS,
    SLIM_TYPES,
    get_assay_display_name,
    get_assay_types,
    get_biosample_names,
    get_biosamples_for_organ,
    get_biosamples_for_slim,
    get_lab_names,
    get_life_stages,
    get_organ_display_name,
    get_organ_systems,
    get_organism_display,
    get_organisms,
    get_primary_organ_for_biosample,
    get_slim_categories,
    get_slim_display_name,
    get_targets,
)
from src.utils.cache import CacheManager
from src.visualization.plots import DimensionalityReducer, PlotGenerator

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="MetaENCODE",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Component Initialization with Caching ---


@st.cache_resource
def get_cache_manager() -> CacheManager:
    """Get or create the cache manager instance."""
    return CacheManager()


@st.cache_resource
def get_api_client() -> EncodeClient:
    """Get or create the API client instance."""
    return EncodeClient()


@st.cache_resource
def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create the embedding generator instance."""
    return EmbeddingGenerator()


@st.cache_resource
def get_metadata_processor() -> MetadataProcessor:
    """Get or create the metadata processor instance."""
    return MetadataProcessor()


@st.cache_resource
def get_feature_combiner() -> FeatureCombiner:
    """Get or create the feature combiner instance."""
    return FeatureCombiner()


@st.cache_data
def load_cached_data(
    _cache_mgr: CacheManager,
) -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    FeatureCombiner | None,
]:
    """Load precomputed metadata, embeddings, and combined vectors from cache.

    Args:
        _cache_mgr: Cache manager instance (prefixed with _ to avoid hashing).

    Returns:
        Tuple of (metadata_df, text_embeddings, combined_vectors, feature_combiner)
        or (None, None, None, None) if not cached.
    """
    if _cache_mgr.exists("metadata") and _cache_mgr.exists("embeddings"):
        metadata = _cache_mgr.load("metadata")
        embeddings = _cache_mgr.load("embeddings")

        # Try to load combined vectors and combiner (Phase 2 data)
        combined_vectors = None
        feature_combiner = None
        if _cache_mgr.exists("combined_vectors"):
            combined_vectors = _cache_mgr.load("combined_vectors")
        if _cache_mgr.exists("feature_combiner"):
            feature_combiner = _cache_mgr.load("feature_combiner")

        return metadata, embeddings, combined_vectors, feature_combiner
    return None, None, None, None


@st.cache_resource
def get_filter_manager() -> SearchFilterManager:
    """Get or create the search filter manager instance."""
    return SearchFilterManager()


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    defaults = {
        "selected_dataset": None,
        "selected_index": None,
        "search_results": None,
        "similar_datasets": None,
        "metadata_df": None,
        "embeddings": None,
        "combined_vectors": None,
        "feature_combiner": None,
        "similarity_engine": None,
        "coords_2d": None,
        # New filter state using FilterState dataclass
        "filter_state": FilterState(),
        # Legacy filter_settings for backward compatibility
        "filter_settings": {
            "organism": None,
            "assay_type": None,
            "top_n": 10,
        },
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> dict:
    """Render sidebar with search and filter controls.

    The sidebar now includes:
    - Assay type selection with autocomplete
    - Organism selection with genome assembly labels
    - Hierarchical biosample selection (body part -> tissue)
    - Histone modification / target selection
    - Age/developmental stage search
    - More Options (lab, replicates)

    Returns:
        Dictionary containing current filter settings.
    """
    filter_mgr = get_filter_manager()

    st.sidebar.title("MetaENCODE")
    st.sidebar.markdown("*ENCODE Dataset Similarity Search*")
    st.sidebar.divider()

    # --- Primary Filters (always visible) ---
    st.sidebar.subheader("Search Filters")
    st.sidebar.caption(
        "Filters apply to Search results only. Similar datasets show pure ML similarity."
    )

    # 1. Free text description search (most general filter - at top)
    description_search = st.sidebar.text_input(
        "Description search",
        value=st.session_state.filter_state.description_search or "",
        placeholder="e.g., 8-week cerebellum, H3K27ac",
        help="Search in experiment descriptions, titles, and metadata",
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
        help="Choose how to classify biosamples (organ, cell type, germ layer, or body system)",
        key="filter_classification_type",
    )
    st.session_state.classification_type = classification_type

    # Dynamic category selector based on classification type
    category_data = get_slim_categories(classification_type)[:20]
    category_options = [""] + [name for name, _ in category_data]
    category_counts = {name: count for name, count in category_data}
    current_bp = st.session_state.filter_state.body_part or ""

    # For backwards compatibility, body_part stores the selected category
    body_part = st.sidebar.selectbox(
        SLIM_TYPES[classification_type]["display_prefix"],
        options=category_options,
        index=(category_options.index(current_bp) if current_bp in category_options else 0),
        format_func=lambda x: (
            f"All {slim_type_labels[classification_type].lower()}s"
            if x == ""
            else f"{get_slim_display_name(classification_type, x)} ({category_counts.get(x, 0):,})"
        ),
        help=SLIM_TYPES[classification_type]["description"],
        key="filter_body_part",
    )

    # Tissue / Cell Type (filtered by selected category)
    if body_part:
        # Get biosamples for selected category from JSON
        biosamples_data = get_biosamples_for_slim(classification_type, body_part)[:50]
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

    # --- Search Button ---
    # Build search query from filters
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

    if st.sidebar.button(
        "Search ENCODE",
        type="primary",
        use_container_width=True,
        disabled=not filter_state.has_any_filter(),
    ):
        if filter_state.has_any_filter():
            with st.spinner("Searching ENCODE..."):
                try:
                    client = get_api_client()

                    # Map organism key to scientific name for API
                    organism_scientific = None
                    if filter_state.organism:
                        org_map = {
                            "human": "Homo sapiens",
                            "mouse": "Mus musculus",
                            "fly": "Drosophila melanogaster",
                            "worm": "Caenorhabditis elegans",
                        }
                        organism_scientific = org_map.get(
                            filter_state.organism, filter_state.organism
                        )

                    # Use fetch_experiments with all available API parameters
                    results = client.fetch_experiments(
                        assay_type=filter_state.assay_type,
                        organism=organism_scientific,
                        biosample=filter_state.biosample,
                        target=filter_state.target,
                        life_stage=filter_state.age_stage,
                        search_term=filter_state.description_search,
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

                    st.session_state.search_results = results
                    st.sidebar.success(f"Found {len(results)} results")
                except Exception as e:
                    st.sidebar.error(f"Search failed: {e}")
        else:
            st.sidebar.warning("Please set at least one filter")

    # Clear filters button
    if st.sidebar.button("Clear Filters", use_container_width=True):
        st.session_state.filter_state = FilterState()
        st.rerun()

    st.sidebar.divider()

    # --- Data Loading Section ---
    st.sidebar.subheader("Data")
    if st.sidebar.button("Load Sample Data", use_container_width=True):
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


def load_sample_data() -> None:
    """Load a sample of ENCODE experiments for demonstration."""
    with st.spinner("Loading sample data from ENCODE API..."):
        try:
            client = get_api_client()
            processor = get_metadata_processor()
            embedder = get_embedding_generator()
            combiner = get_feature_combiner()

            # Fetch a small sample of experiments
            raw_df = client.fetch_experiments(limit=100)

            if raw_df.empty:
                st.error("No experiments found")
                return

            # Process metadata
            processed_df = processor.process(raw_df)

            # Validate records and filter invalid ones
            valid_mask = processed_df.apply(
                lambda row: processor.validate_record(row.to_dict()), axis=1
            )
            if not valid_mask.all():
                invalid_count = (~valid_mask).sum()
                st.warning(
                    f"Filtered {invalid_count} records missing required metadata"
                )
                processed_df = processed_df[valid_mask].reset_index(drop=True)

            if processed_df.empty:
                st.error("No valid experiments found after validation")
                return

            # Generate text embeddings
            st.info("Generating text embeddings...")
            texts = processed_df["combined_text"].tolist()
            text_embeddings = embedder.encode(texts, show_progress=False)

            # Fit feature combiner and generate combined vectors
            st.info("Combining features (text + categorical + numeric)...")
            combiner.fit(processed_df, text_embedding_dim=text_embeddings.shape[1])
            combined_vectors = combiner.transform(processed_df, text_embeddings)

            # Fit similarity engine with COMBINED vectors (not text-only)
            similarity_engine = SimilarityEngine()
            similarity_engine.fit(combined_vectors)

            # Store in session state
            st.session_state.metadata_df = processed_df
            st.session_state.embeddings = text_embeddings
            st.session_state.combined_vectors = combined_vectors
            st.session_state.feature_combiner = combiner
            st.session_state.similarity_engine = similarity_engine

            # Cache the data (only if not overwriting a larger existing cache)
            cache_mgr = get_cache_manager()
            existing_meta = (
                cache_mgr.load("metadata") if cache_mgr.exists("metadata") else None
            )

            if existing_meta is not None and len(existing_meta) > len(processed_df):
                st.warning(
                    f"Skipped caching: existing cache has "
                    f"{len(existing_meta)} experiments, "
                    f"not overwriting with {len(processed_df)} samples. "
                    "Use precompute_embeddings.py to update full cache."
                )
            else:
                cache_mgr.save("metadata", processed_df)
                cache_mgr.save("embeddings", text_embeddings)
                cache_mgr.save("combined_vectors", combined_vectors)
                cache_mgr.save("feature_combiner", combiner)

            # Show feature breakdown
            breakdown = combiner.get_feature_breakdown()
            text_dim = breakdown.get("text_embedding", 0)
            numeric_dim = breakdown.get("numeric_features", 0)
            categorical_dim = sum(
                v
                for k, v in breakdown.items()
                if k not in ["text_embedding", "numeric_features"]
            )
            st.success(
                f"Loaded {len(processed_df)} experiments with "
                f"{combiner.feature_dim}-dim combined vectors "
                f"(text: {text_dim}, categorical: {categorical_dim}, "
                f"numeric: {numeric_dim})"
            )

        except Exception as e:
            st.error(f"Failed to load data: {e}")


def render_main_content() -> None:
    """Render main content area."""
    st.title("MetaENCODE")
    st.markdown(
        "**Discover related ENCODE datasets through "
        "metadata-driven similarity scoring**"
    )

    # Tabs for different views
    tab_search, tab_similar, tab_visualize = st.tabs(
        ["Search & Select", "Similar Datasets", "Visualize"]
    )

    with tab_search:
        render_search_tab()

    with tab_similar:
        render_similar_tab()

    with tab_visualize:
        render_visualize_tab()


def render_search_tab() -> None:
    """Render the search and selection tab."""
    st.header("Search & Select Dataset")

    # Get current filter state
    filter_state = st.session_state.filter_state
    max_results = filter_state.max_results

    # Display search results if available
    if st.session_state.search_results is not None:
        results_df = st.session_state.search_results

        if not results_df.empty:
            st.subheader(f"Search Results ({len(results_df)} datasets)")

            # Display as interactive table with formatted columns
            display_cols = [
                "accession",
                "assay_term_name",
                "organism",
                "biosample_term_name",
                "description",
            ]
            display_cols = [c for c in display_cols if c in results_df.columns]

            # Create display DataFrame with formatting
            display_df = results_df[display_cols].copy()

            # Format organism with genome assembly
            if "organism" in display_df.columns:
                display_df["organism"] = display_df["organism"].apply(
                    format_organism_display
                )

            # Truncate descriptions for display
            if "description" in display_df.columns:
                display_df["description"] = display_df["description"].apply(
                    lambda x: (str(x)[:80] + "...") if len(str(x)) > 80 else str(x)
                )

            # Rename columns for display
            column_labels = {
                "accession": "Accession",
                "assay_term_name": "Assay",
                "organism": "Organism [Assembly]",
                "biosample_term_name": "Biosample",
                "description": "Description",
            }
            display_df = display_df.rename(
                columns={
                    k: v for k, v in column_labels.items() if k in display_df.columns
                }
            )

            # Let user select a row
            selection = st.dataframe(
                display_df.head(max_results),
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
            )

            # Handle selection
            if selection and selection.selection.rows:
                selected_idx = selection.selection.rows[0]
                selected_row = results_df.iloc[selected_idx]
                st.session_state.selected_dataset = selected_row.to_dict()
                st.success(f"Selected: {selected_row['accession']}")

            # Show info about filters applied
            if filter_state.has_any_filter():
                active_filters = []
                if filter_state.assay_type:
                    active_filters.append(f"Assay: {filter_state.assay_type}")
                if filter_state.organism:
                    active_filters.append(
                        f"Organism: {format_organism_display(filter_state.organism)}"
                    )
                if filter_state.target:
                    active_filters.append(f"Target: {filter_state.target}")
                if filter_state.biosample:
                    active_filters.append(f"Biosample: {filter_state.biosample}")
                if filter_state.age_stage:
                    active_filters.append(f"Stage: {filter_state.age_stage}")
                if active_filters:
                    st.caption(f"Filtered by: {' | '.join(active_filters)}")
        else:
            st.info("No results found. Try adjusting your filters.")
    else:
        st.info(
            "Use the filters in the sidebar to search for datasets, "
            "or enter an accession number below."
        )

    st.divider()

    # Manual accession input
    st.subheader("Or enter an accession directly")
    accession = st.text_input(
        "ENCODE Accession",
        placeholder="e.g., ENCSR000AKS",
        help="Enter an ENCODE experiment accession number",
    )

    if st.button("Load Dataset"):
        if accession.strip():
            with st.spinner(f"Loading {accession}..."):
                try:
                    client = get_api_client()
                    dataset = client.fetch_experiment_by_accession(accession.strip())
                    st.session_state.selected_dataset = dataset
                    st.success(f"Loaded dataset: {accession}")
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Failed to load dataset: {e}")
        else:
            st.warning("Please enter an accession number")

    # Display selected dataset
    if st.session_state.selected_dataset is not None:
        st.divider()
        st.subheader("Selected Dataset")
        dataset = st.session_state.selected_dataset

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accession", dataset.get("accession", "N/A"))
            st.metric("Assay", dataset.get("assay_term_name", "N/A"))
        with col2:
            st.metric(
                "Organism",
                format_organism_display(dataset.get("organism", "")),
            )
            st.metric("Biosample", dataset.get("biosample_term_name", "N/A"))

        with st.expander("Full Metadata"):
            st.json(dataset)


def format_organism_display(organism: str) -> str:
    """Format organism name with genome assembly label.

    Delegates to get_organism_display for consistent formatting
    across all organisms, including those not in the known list.

    Args:
        organism: Organism name (common or scientific).

    Returns:
        Formatted string with assembly (e.g., "Human [hg38]") or
        just the organism name if no assembly info available.
    """
    if not organism:
        return "N/A"
    return get_organism_display(organism)


def render_similar_tab() -> None:
    """Render the similar datasets tab."""
    st.header("Similar Datasets")

    if st.session_state.selected_dataset is None:
        st.info("Select a dataset first to find similar experiments.")
        return

    # Check if we have loaded data
    if st.session_state.metadata_df is None or st.session_state.embeddings is None:
        st.warning(
            "Please load sample data first using the 'Load Sample Data' button "
            "in the sidebar."
        )
        return

    selected = st.session_state.selected_dataset
    st.write(f"Finding datasets similar to: **{selected.get('accession', 'Unknown')}**")

    # Get filter state
    filter_state = st.session_state.filter_state
    top_n = filter_state.max_results

    if st.button("Find Similar Datasets", type="primary"):
        with st.spinner("Computing similarities..."):
            try:
                embedder = get_embedding_generator()
                similarity_engine = st.session_state.similarity_engine
                feature_combiner = st.session_state.feature_combiner

                if similarity_engine is None:
                    st.error(
                        "Similarity engine not initialized. Please load data first."
                    )
                    return

                # Generate text embedding for selected dataset
                text = f"{selected.get('description', '')} {selected.get('title', '')}"
                text_embedding = embedder.encode_single(text)

                # Generate combined query vector (if feature combiner is available)
                if feature_combiner is not None and feature_combiner.is_fitted:
                    query_vector = feature_combiner.transform_single(
                        selected, text_embedding
                    )
                else:
                    # Fallback to text-only similarity
                    query_vector = text_embedding

                # Find more similar datasets than requested (for post-filtering)
                fetch_n = max(top_n * 3, 30)
                similar_df = similarity_engine.find_similar(
                    query_vector, n=fetch_n, exclude_self=True
                )

                # Get metadata for similar datasets
                metadata_df = st.session_state.metadata_df
                results = []
                for _, row in similar_df.iterrows():
                    idx = int(row["index"])
                    if idx < len(metadata_df):
                        meta = metadata_df.iloc[idx].to_dict()
                        meta["similarity_score"] = row["similarity_score"]
                        results.append(meta)

                st.session_state.similar_datasets = pd.DataFrame(results)

            except Exception as e:
                st.error(f"Error computing similarities: {e}")

    # Display similar datasets
    if st.session_state.similar_datasets is not None:
        similar = st.session_state.similar_datasets

        if not similar.empty:
            st.subheader("Most Similar Datasets")

            # Limit to max_results (no filtering - pure similarity ranking)
            display_similar = similar.head(top_n)

            # Display columns with proper formatting
            display_cols = [
                "similarity_score",
                "accession",
                "assay_term_name",
                "organism",
                "biosample_term_name",
                "description",
            ]
            display_cols = [c for c in display_cols if c in display_similar.columns]

            display_df = display_similar[display_cols].copy()

            # Format similarity score
            display_df["similarity_score"] = display_df["similarity_score"].apply(
                lambda x: f"{x:.3f}"
            )

            # Format organism with assembly
            if "organism" in display_df.columns:
                display_df["organism"] = display_df["organism"].apply(
                    format_organism_display
                )

            # Truncate description
            if "description" in display_df.columns:
                display_df["description"] = display_df["description"].apply(
                    lambda x: (str(x)[:60] + "...") if len(str(x)) > 60 else str(x)
                )

            # Rename columns for display
            column_labels = {
                "similarity_score": "Similarity",
                "accession": "Accession",
                "assay_term_name": "Assay",
                "organism": "Organism [Assembly]",
                "biosample_term_name": "Biosample",
                "description": "Description",
            }
            display_df = display_df.rename(
                columns={
                    k: v for k, v in column_labels.items() if k in display_df.columns
                }
            )

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Link to ENCODE
            st.markdown("Click accession numbers to view on ENCODE portal:")
            for _, row in display_similar.head(5).iterrows():
                acc = row.get("accession", "")
                if acc:
                    url = f"https://www.encodeproject.org/experiments/{acc}/"
                    st.markdown(f"- [{acc}]({url})")


def render_visualize_tab() -> None:
    """Render the visualization tab."""
    st.header("Dataset Visualization")

    if st.session_state.metadata_df is None or st.session_state.embeddings is None:
        st.info(
            "Load sample data first using the 'Load Sample Data' button "
            "in the sidebar to visualize datasets."
        )
        return

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Options")

        reduction_method = st.selectbox(
            "Reduction Method",
            options=["pca", "umap"],
            index=0,
            help="PCA is faster, UMAP preserves local structure better",
        )

        # Determine available color options based on metadata columns
        available_colors = ["assay_term_name", "organism", "lab"]
        if st.session_state.metadata_df is not None:
            # Add slim type color options if columns exist
            slim_color_columns = [
                "organ",
                "cell_type",
                "developmental_layer",
                "body_system",
            ]
            for col in slim_color_columns:
                if col in st.session_state.metadata_df.columns:
                    available_colors.insert(-1, col)  # Insert before "lab"

        color_display_names = {
            "assay_term_name": "Assay Type",
            "organism": "Organism",
            "organ": "Organ System",
            "cell_type": "Cell Type",
            "developmental_layer": "Germ Layer",
            "body_system": "Body System",
            "lab": "Lab",
        }
        color_option = st.selectbox(
            "Color By",
            options=available_colors,
            format_func=lambda x: color_display_names.get(
                x, x.replace("_", " ").title()
            ),
        )

        if st.button("Generate Visualization", type="primary"):
            generate_visualization(reduction_method, color_option)

    with col1:
        st.subheader("Embedding Space")

        if st.session_state.coords_2d is not None:
            metadata_df = st.session_state.metadata_df
            coords = st.session_state.coords_2d

            # Get highlight indices if we have similar datasets
            highlight_idx = None
            if st.session_state.similar_datasets is not None:
                # Find indices of similar datasets in the full metadata
                similar_accs = set(
                    st.session_state.similar_datasets["accession"].tolist()
                )
                highlight_idx = [
                    i
                    for i, acc in enumerate(metadata_df["accession"])
                    if acc in similar_accs
                ]

            # Generate plot
            plotter = PlotGenerator(reduction_method=reduction_method)
            fig = plotter.scatter_plot(
                coords,
                metadata_df,
                color_by=color_option,
                title="Dataset Similarity Map",
                highlight_indices=highlight_idx,
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "Click 'Generate Visualization' to create the embedding plot. "
                "This may take a moment for UMAP."
            )


def generate_visualization(method: str, color_by: str) -> None:
    """Generate 2D visualization of embeddings.

    Args:
        method: Dimensionality reduction method ('pca' or 'umap').
        color_by: Column to color points by.
    """
    with st.spinner(f"Computing {method.upper()} projection..."):
        try:
            embeddings = st.session_state.embeddings
            reducer = DimensionalityReducer(method=method)
            coords_2d = reducer.fit_transform(embeddings)
            st.session_state.coords_2d = coords_2d
        except Exception as e:
            st.error(f"Error generating visualization: {e}")


def main() -> None:
    """Main application entry point."""
    # Initialize session state
    init_session_state()

    # Try to load cached data on startup
    cache_mgr = get_cache_manager()
    if st.session_state.metadata_df is None:
        cached_meta, cached_emb, cached_combined, cached_combiner = load_cached_data(
            cache_mgr
        )
        if cached_meta is not None and cached_emb is not None:
            st.session_state.metadata_df = cached_meta
            st.session_state.embeddings = cached_emb

            # Use combined vectors if available, otherwise fallback to text embeddings
            if cached_combined is not None:
                st.session_state.combined_vectors = cached_combined
                similarity_engine = SimilarityEngine()
                similarity_engine.fit(cached_combined)
            else:
                similarity_engine = SimilarityEngine()
                similarity_engine.fit(cached_emb)

            st.session_state.similarity_engine = similarity_engine

            # Restore feature combiner if available
            if cached_combiner is not None:
                st.session_state.feature_combiner = cached_combiner

    # Render sidebar and get filter settings
    filters = render_sidebar()

    # Update session state with filter settings
    st.session_state.filter_settings.update(filters)

    # Render main content
    render_main_content()


if __name__ == "__main__":
    main()
