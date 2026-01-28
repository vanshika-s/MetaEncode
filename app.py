# app.py
"""MetaENCODE: ENCODE Dataset Similarity Search Application.

This Streamlit application enables researchers to discover related ENCODE
datasets through metadata-driven similarity scoring. Users can search for
datasets, select a seed dataset, and explore similar experiments through
interactive visualizations.

Run with: streamlit run app.py
"""

import streamlit as st

from src.ui.components.session import init_session_state, load_cached_data_into_session
from src.ui.sidebar import render_sidebar
from src.ui.tabs import render_search_tab, render_similar_tab, render_visualize_tab

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="MetaENCODE",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_main_content() -> None:
    """Render main content area with tabs."""
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
        render_visualize_tab()  # Comment out to disable visualization


def main() -> None:
    """Main application entry point."""
    # Initialize session state
    init_session_state()

    # Load cached data into session state (if available)
    load_cached_data_into_session()

    # Render sidebar and get filter settings
    filters = render_sidebar()

    # Update session state with filter settings
    st.session_state.filter_settings.update(filters)

    # Render main content
    render_main_content()


if __name__ == "__main__":
    main()
