# src/utils/history.py
"""Persistent selection history for recently loaded datasets.

Stores recent dataset selections in a JSON file alongside the cache directory,
enabling users to quickly recall previously loaded datasets across sessions.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class SelectionHistory:
    """Manage persistent history of dataset selections.

    Stores entries as a JSON array ordered by most-recently-used (MRU).
    Re-selecting an existing accession moves it to the top without duplicating.
    Bounded to MAX_ENTRIES to prevent unbounded growth.

    Example:
        >>> history = SelectionHistory()
        >>> history.add("ENCSR000AKS", {"accession": "ENCSR000AKS", ...})
        >>> entries = history.get_entries()
        >>> label = SelectionHistory.format_entry_label(entries[0])
    """

    DEFAULT_PATH = Path("data/cache/selection_history.json")
    MAX_ENTRIES = 30

    def __init__(self, path: Optional[str] = None) -> None:
        """Initialize selection history manager.

        Args:
            path: Path to the JSON history file. Defaults to data/cache/selection_history.json.
        """
        self.path = Path(path) if path else self.DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict[str, Any]] = []
        self.load()

    def load(self) -> list[dict[str, Any]]:
        """Load history entries from disk.

        Returns:
            List of history entry dicts, most recent first.
        """
        if not self.path.exists():
            self._entries = []
            return self._entries

        try:
            text = self.path.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, list):
                self._entries = data
            else:
                self._entries = []
        except (json.JSONDecodeError, OSError):
            self._entries = []

        return self._entries

    def save(self) -> None:
        """Persist current entries to disk using atomic write."""
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), suffix=".tmp"
        )
        tmp_path = Path(tmp_name)
        try:
            with open(fd, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, indent=2, ensure_ascii=False)
            tmp_path.rename(self.path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def add(self, accession: str, dataset: dict[str, Any]) -> list[dict[str, Any]]:
        """Add or promote a dataset selection to the top of history.

        If the accession already exists, it is moved to the top (dedup).
        The list is bounded to MAX_ENTRIES.

        Args:
            accession: ENCODE experiment accession (e.g., "ENCSR000AKS").
            dataset: Full dataset metadata dict from the API.

        Returns:
            Updated list of history entries.
        """
        # Remove existing entry for this accession (dedup)
        self._entries = [
            e for e in self._entries if e.get("accession") != accession
        ]

        description = str(dataset.get("description", ""))
        if len(description) > 120:
            description = description[:117] + "..."

        entry = {
            "accession": accession,
            "assay_term_name": dataset.get("assay_term_name", ""),
            "biosample_term_name": dataset.get("biosample_term_name", ""),
            "organism": dataset.get("organism", ""),
            "description": description,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Insert at front (MRU ordering)
        self._entries.insert(0, entry)

        # Enforce max entries
        self._entries = self._entries[: self.MAX_ENTRIES]

        self.save()
        return self._entries

    def get_entries(self) -> list[dict[str, Any]]:
        """Return current history entries, most recent first."""
        return list(self._entries)

    def clear(self) -> None:
        """Remove all history entries and delete the file."""
        self._entries = []
        if self.path.exists():
            self.path.unlink()

    @staticmethod
    def format_entry_label(entry: dict[str, Any]) -> str:
        """Format a history entry as a human-readable dropdown label.

        Args:
            entry: A history entry dict.

        Returns:
            Formatted string, e.g. "ENCSR000AKS - ChIP-seq | K562 | Homo sapiens"
        """
        accession = entry.get("accession", "unknown")
        parts = [accession]

        detail_parts = []
        if entry.get("assay_term_name"):
            detail_parts.append(entry["assay_term_name"])
        if entry.get("biosample_term_name"):
            detail_parts.append(entry["biosample_term_name"])
        if entry.get("organism"):
            detail_parts.append(entry["organism"])

        if detail_parts:
            parts.append(" | ".join(detail_parts))

        return " - ".join(parts)
