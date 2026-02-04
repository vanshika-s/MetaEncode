# src/ui/formatters.py
"""Display formatting utilities for MetaENCODE UI.

This module provides formatting functions for displaying data in the UI,
such as organism names with genome assemblies and ENCODE URL generation.
"""

from src.ui.vocabularies import get_organism_display

# ENCODE Portal base URL
ENCODE_BASE_URL = "https://www.encodeproject.org"


def get_encode_experiment_url(accession: str) -> str:
    """Generate ENCODE experiment URL from accession.

    Args:
        accession: ENCODE accession ID (e.g., ENCSR000AKS).

    Returns:
        Full URL to the experiment page on ENCODE portal.
    """
    if not accession:
        return ""
    return f"{ENCODE_BASE_URL}/experiments/{accession}/"


def format_accession_as_link(accession: str) -> str:
    """Format accession as markdown hyperlink to ENCODE portal.

    Args:
        accession: ENCODE accession ID (e.g., ENCSR000AKS).

    Returns:
        Markdown link string (e.g., "[ENCSR000AKS](https://...)").
    """
    if not accession:
        return "N/A"
    url = get_encode_experiment_url(accession)
    return f"[{accession}]({url})"


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
