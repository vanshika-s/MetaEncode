# src/ui/tabs/search.py
"""Search and dataset selection tab for MetaENCODE."""

import streamlit as st

from src.ui.components.initializers import get_api_client
from src.ui.formatters import format_organism_display, truncate_text


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
                    lambda x: truncate_text(str(x), 80)
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
