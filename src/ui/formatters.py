# src/ui/formatters.py
"""Display formatting utilities for MetaENCODE UI.

This module provides formatting functions for displaying data in the UI,
such as organism names with genome assemblies.
"""

from src.ui.vocabularies import get_organism_display


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


def truncate_text(text: str, max_length: int = 80) -> str:
    """Truncate text to specified length with ellipsis.

    Args:
        text: Text to truncate.
        max_length: Maximum length before truncation.

    Returns:
        Truncated text with '...' suffix if needed.
    """
    text_str = str(text) if text else ""
    if len(text_str) > max_length:
        return text_str[:max_length] + "..."
    return text_str
