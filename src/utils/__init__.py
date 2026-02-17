# src/utils/__init__.py
"""Utility functions and caching module."""

from .cache import CacheManager
from .history import SelectionHistory

# Note: data_loader is intentionally not imported here to avoid circular imports
# and to keep data loading separate from the core utils package. Import any
# data loading functionality directly from its own module in the UI or
# application layer as needed, rather than exposing it via src.utils.

# Lazy imports for optional spell check module
# (requires symspellpy and jellyfish)
try:
    from .spell_check import (
        SpellingSuggestion,
        VocabularySpellChecker,
        correct_spelling,
        get_spell_checker,
        suggest_correction,
    )

    _SPELL_CHECK_AVAILABLE = True
except ImportError:
    _SPELL_CHECK_AVAILABLE = False
    VocabularySpellChecker = None
    SpellingSuggestion = None
    get_spell_checker = None
    suggest_correction = None
    correct_spelling = None

__all__ = [
    "CacheManager",
    "SelectionHistory",
    "VocabularySpellChecker",
    "SpellingSuggestion",
    "get_spell_checker",
    "suggest_correction",
    "correct_spelling",
]
