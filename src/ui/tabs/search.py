# src/ui/tabs/search.py
"""Search and dataset selection tab for MetaENCODE."""

from typing import Any

import streamlit as st

from src.ui.components.initializers import get_api_client, get_selection_history
from src.ui.formatters import (
    format_accession_as_link,
    format_organism_display,
    get_encode_experiment_url,
    truncate_text,
)
from src.utils.history import SelectionHistory


def _save_to_history(dataset: dict[str, Any]) -> None:
    """Save a dataset selection to persistent history and sync session state."""
    accession = dataset.get("accession", "")
    if not accession:
        return
    history = get_selection_history()
    entries = history.add(accession, dataset)
    st.session_state.selection_history = entries


def render_search_tab() -> None:
    """Render the search and selection tab."""
    st.markdown(
        """
        <style>
        .search-title {
            font-size: 1.9rem;
            font-weight: 650;
            margin-bottom: 0.25rem;
        }

        .search-subtitle {
            font-size: 1.35rem;
            font-   weight: 600;
            margin-top: 1.75rem;
            margin-bottom: 0.4rem;
        }

        .search-helper {
            font-size: 1.1rem;
            font-weight: 500;
            color: #666;
            margin-top: 1.2rem;
            margin-bottom: 0.3rem;
        }
        
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        "<div class='search-title'>Search & Select Dataset</div>",
        unsafe_allow_html=True,
    )
    

    # Get current filter state
    filter_state = st.session_state.filter_state
    max_results = filter_state.max_results

    # Display search results if available
    if st.session_state.search_results is not None:
        results_df = st.session_state.search_results

        if not results_df.empty:
            st.markdown(
                f"<div class='search-subtitle'>Search Results ({len(results_df)} datasets)</div>",
                unsafe_allow_html=True,
            )
            
            st.markdown("Please pick one of the datasets below.")
            
            

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

            # Replace accession values with ENCODE URLs for clickable links
            display_df["accession"] = results_df["accession"].apply(get_encode_experiment_url)

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

            # Configure Accession column as clickable link showing accession ID
            column_config = {
                "Accession": st.column_config.LinkColumn(
                    "Accession",
                    display_text=r"experiments/(ENC[^/]+)/",
                    help="Click to open on ENCODE Portal",
                ),
            }

            # Let user select a row
            selection = st.dataframe(
                display_df.head(max_results),
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                column_config=column_config,
            )

            # Handle selection
            previous_selection_index = st.session_state.get("previous_selection_index")
            if selection and getattr(selection, "selection", None) is not None:
                if selection.selection.rows:
                    selected_idx = selection.selection.rows[0]
                    if selected_idx != previous_selection_index:
                        st.session_state.previous_selection_index = selected_idx
                        selected_row = results_df.iloc[selected_idx]
                        st.session_state.selected_dataset = selected_row.to_dict()
                        _save_to_history(selected_row.to_dict())
                        st.success(f"Selected: {selected_row['accession']}. Scroll down to see info.")
                else:
                    # No rows currently selected; clear previous index so a future selection is detected.
                    st.session_state.previous_selection_index = None

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

    # Manual accession input with recent selections
    st.subheader("Or enter an accession directly")

    # Recent selections dropdown (only shown when history exists)
    history_entries = st.session_state.get("selection_history", [])
    if history_entries:
        labels = ["Select a recent dataset..."] + [
            SelectionHistory.format_entry_label(e) for e in history_entries
        ]
        history_choice = st.selectbox(
            "Recent selections",
            options=range(len(labels)),
            format_func=lambda i: labels[i],
            key="history_selectbox",
        )

        # Load from history when a real entry is selected (not placeholder)
        last_choice = st.session_state.get("_last_history_selection", 0)
        if history_choice and history_choice != last_choice:
            st.session_state._last_history_selection = history_choice
            selected_entry = history_entries[history_choice - 1]
            selected_accession = selected_entry["accession"]
            with st.spinner(f"Loading {selected_accession}..."):
                try:
                    client = get_api_client()
                    dataset = client.fetch_experiment_by_accession(selected_accession)
                    st.session_state.selected_dataset = dataset
                    _save_to_history(dataset)
                    st.success(f"Loaded dataset: {selected_accession}. Scroll down to see info.")
                except (ValueError, Exception) as e:
                    st.error(f"Failed to load dataset: {e}")

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
                    _save_to_history(dataset)
                    st.success(f"Loaded dataset: {accession}. Scroll down to see info.")
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Failed to load dataset: {e}")
        else:
            st.warning("Please enter an accession number")

    # Display selected dataset
    
    if st.session_state.selected_dataset is not None:
        st.divider()
        st.markdown(
            "<div class='search-subtitle'>Selected Dataset</div>",
            unsafe_allow_html=True,
        )
        dataset = st.session_state.selected_dataset

        col1, col2 = st.columns(2)
        with col1:
            # Accession as clickable link to ENCODE portal
            accession = dataset.get("accession", "N/A")
            accession_link = format_accession_as_link(accession)
            st.markdown(f"**Accession:** {accession_link}")
            st.metric("Assay", dataset.get("assay_term_name", "N/A"))
        with col2:
            st.metric(
                "Organism",
                format_organism_display(dataset.get("organism", "")),
            )
            st.metric("Biosample", dataset.get("biosample_term_name", "N/A"))

        with st.expander("Full Metadata"):
            st.json(dataset)
