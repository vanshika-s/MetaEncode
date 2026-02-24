# app.py
"""MetaENCODE: ENCODE Dataset Similarity Search Application.

This Streamlit application enables researchers to discover related ENCODE
datasets through metadata-driven similarity scoring. Users can search for
datasets, select a seed dataset, and explore similar experiments through
interactive visualizations.

Run with: streamlit run app.py
"""

import streamlit as st

from src.ui.components.session import (
    init_session_state,
    load_cached_data_into_session,
    load_selection_history_into_session,
)
from src.ui.sidebar import render_sidebar
from src.ui.tabs import render_search_tab, render_similar_tab, render_visualize_tab

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="MetaENCODE",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
    <style>
        /* Margins */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            padding-left: 4rem;
            padding-right: 4rem;
        }
        
        /* Title */
        .title {
            font-size: 3.0rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
    </style>
""", unsafe_allow_html=True)


def render_main_content() -> None:
    """Render main content area with cards"""
    st.markdown(
        "<div class='title'>MetaENCODE</div>",
        unsafe_allow_html=True,
    )
    st.markdown("**Discover related ENCODE datasets and visualize dataset similarity.**")

    # Styling for buttons 
    st.markdown("""
        <style>

        [data-testid="stVerticalBlock"] > div:has(div.card-container) {
            margin-top: -10px !important;
        }
        
        /* Base Card */
        [data-testid="stVerticalBlock"] > div:has(div.card-container) button {
            height: 4.3rem;
            border-radius: 8px;
            border: 2px solid #afbc88;
            background-color: #ffffff;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: all 0.2s ease-in-out;
            color: #31333F;
        }
        
        hr {
            margin-top: 3px !important;
            margin-bottom: 5px !important;
        }
        
        /* Button Text */
        [data-testid="stVerticalBlock"] > div:has(div.card-container) button p {
            font-size: 1.4rem;
            font-weight: 400;
        }

        /* Hover Effect */
        [data-testid="stVerticalBlock"] > div:has(div.card-container) button:hover {
            border-color: #618B4A;
            color: #618B4A;
            transform: translateY(-3px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # Track state
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Search"

    # Card container
    with st.container():
        st.markdown('<div class="card-container"></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Search & Select", use_container_width=True, key="btn_search"):
                st.session_state.active_tab = "Search"
                st.rerun()
            if st.session_state.active_tab == "Search":
                st.markdown("<div style='border-bottom: 5px solid #618B4A; margin-top: -15px;'></div>", unsafe_allow_html=True)

        with col2:
            if st.button("Similar Datasets", use_container_width=True, key="btn_similar"):
                st.session_state.active_tab = "Similar"
                st.rerun()
            if st.session_state.active_tab == "Similar":
                st.markdown("<div style='border-bottom: 5px solid #618B4A; margin-top: -15px;'></div>", unsafe_allow_html=True)

        with col3:
            if st.button("Visualize", use_container_width=True, key="btn_visualize"):
                st.session_state.active_tab = "Visualize"
                st.rerun()
            if st.session_state.active_tab == "Visualize":
                st.markdown("<div style='border-bottom: 5px solid #618B4A; margin-top: -15px;'></div>", unsafe_allow_html=True)

    st.divider()

    # Display content
    if st.session_state.active_tab == "Search":
       render_search_tab()
    elif st.session_state.active_tab == "Similar":
       render_similar_tab()
    elif st.session_state.active_tab == "Visualize":
       render_visualize_tab()




def main() -> None:
    """Main application entry point."""
    # Initialize session state
    init_session_state()

    # Load cached data into session state (if available)
    load_cached_data_into_session()

    # Load selection history from disk
    load_selection_history_into_session()

    # Render sidebar and get filter settings
    filters = render_sidebar()

    # Update session state with filter settings
    st.session_state.filter_settings.update(filters)

    # Render main content
    render_main_content()


if __name__ == "__main__":
    main()
    