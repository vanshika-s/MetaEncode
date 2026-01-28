# src/utils/__init__.py
"""Utility functions and caching module."""

from .cache import CacheManager

# Note: data_loader is NOT imported here to avoid circular imports.
# Import directly: from src.utils.data_loader import load_sample_data

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
    "VocabularySpellChecker",
    "SpellingSuggestion",
    "get_spell_checker",
    "suggest_correction",
    "correct_spelling",
]
