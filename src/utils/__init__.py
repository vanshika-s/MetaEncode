# src/utils/__init__.py
"""Utility functions and caching module."""

from .cache import CacheManager

# Lazy imports for optional spell check module
# (requires symspellpy and jellyfish)
try:
    from .spell_check import (
        VocabularySpellChecker,
        SpellingSuggestion,
        get_spell_checker,
        suggest_correction,
        correct_spelling,
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
