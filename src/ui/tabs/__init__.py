# src/ui/tabs/__init__.py
"""Tab components for MetaENCODE main content area."""

from src.ui.tabs.search import render_search_tab
from src.ui.tabs.similar import render_similar_tab
from src.ui.tabs.visualize import render_visualize_tab

__all__ = [
    "render_search_tab",
    "render_similar_tab",
    "render_visualize_tab",
]
