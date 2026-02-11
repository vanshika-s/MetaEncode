# tests/test_utils/test_history.py
"""Tests for SelectionHistory class."""

import json
from pathlib import Path

import pytest

from src.utils.history import SelectionHistory


SAMPLE_DATASET = {
    "accession": "ENCSR000AKS",
    "assay_term_name": "ChIP-seq",
    "biosample_term_name": "K562",
    "organism": "Homo sapiens",
    "description": "CTCF ChIP-seq on human K562 cell line",
}

SAMPLE_DATASET_2 = {
    "accession": "ENCSR000BLZ",
    "assay_term_name": "RNA-seq",
    "biosample_term_name": "GM12878",
    "organism": "Homo sapiens",
    "description": "Total RNA-seq on human GM12878",
}

SAMPLE_DATASET_3 = {
    "accession": "ENCSR000CCC",
    "assay_term_name": "ATAC-seq",
    "biosample_term_name": "HepG2",
    "organism": "Mus musculus",
    "description": "ATAC-seq on mouse HepG2 cell line",
}


@pytest.fixture
def history_path(tmp_path: Path) -> Path:
    """Return path for a temp history JSON file."""
    return tmp_path / "selection_history.json"


@pytest.fixture
def history(history_path: Path) -> SelectionHistory:
    """Create a SelectionHistory with a temp file path."""
    return SelectionHistory(path=str(history_path))


class TestSelectionHistoryInit:
    """Tests for SelectionHistory initialization."""

    def test_init_creates_parent_dir(self, tmp_path: Path) -> None:
        """Parent directory is created if it doesn't exist."""
        nested = tmp_path / "a" / "b" / "history.json"
        SelectionHistory(path=str(nested))
        assert nested.parent.exists()

    def test_init_with_default_path(self) -> None:
        """Default path is data/cache/selection_history.json."""
        h = SelectionHistory.__new__(SelectionHistory)
        # Don't call __init__ to avoid side effects; just check the class attr
        assert SelectionHistory.DEFAULT_PATH == Path("data/cache/selection_history.json")

    def test_init_loads_empty_when_no_file(self, history: SelectionHistory) -> None:
        """Empty entries when no file exists on disk."""
        assert history.get_entries() == []

    def test_max_entries_constant(self) -> None:
        """MAX_ENTRIES is 30."""
        assert SelectionHistory.MAX_ENTRIES == 30


class TestSelectionHistoryLoadSave:
    """Tests for load/save roundtrip."""

    def test_save_and_load_roundtrip(
        self, history: SelectionHistory, history_path: Path
    ) -> None:
        """Entries survive a save/load cycle."""
        history.add("ENCSR000AKS", SAMPLE_DATASET)
        # Create a fresh instance pointing at the same file
        h2 = SelectionHistory(path=str(history_path))
        entries = h2.get_entries()
        assert len(entries) == 1
        assert entries[0]["accession"] == "ENCSR000AKS"

    def test_save_creates_valid_json(
        self, history: SelectionHistory, history_path: Path
    ) -> None:
        """Saved file is valid JSON."""
        history.add("ENCSR000AKS", SAMPLE_DATASET)
        data = json.loads(history_path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 1

    def test_atomic_write_no_tmp_files(
        self, history: SelectionHistory, history_path: Path
    ) -> None:
        """No .tmp files remain after save."""
        history.add("ENCSR000AKS", SAMPLE_DATASET)
        tmp_files = list(history_path.parent.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_corrupted_file_recovery(self, history_path: Path) -> None:
        """Corrupted JSON file is handled gracefully."""
        history_path.write_text("NOT VALID JSON {{{{", encoding="utf-8")
        h = SelectionHistory(path=str(history_path))
        assert h.get_entries() == []

    def test_non_list_json_recovery(self, history_path: Path) -> None:
        """JSON file containing a non-list value resets to empty."""
        history_path.write_text('{"key": "value"}', encoding="utf-8")
        h = SelectionHistory(path=str(history_path))
        assert h.get_entries() == []


class TestSelectionHistoryAdd:
    """Tests for add method."""

    def test_add_single_entry(self, history: SelectionHistory) -> None:
        """Adding one entry returns it in the list."""
        entries = history.add("ENCSR000AKS", SAMPLE_DATASET)
        assert len(entries) == 1
        assert entries[0]["accession"] == "ENCSR000AKS"
        assert entries[0]["assay_term_name"] == "ChIP-seq"
        assert entries[0]["biosample_term_name"] == "K562"
        assert entries[0]["organism"] == "Homo sapiens"

    def test_add_preserves_mru_order(self, history: SelectionHistory) -> None:
        """Most recently added entry is first."""
        history.add("ENCSR000AKS", SAMPLE_DATASET)
        history.add("ENCSR000BLZ", SAMPLE_DATASET_2)
        entries = history.get_entries()
        assert entries[0]["accession"] == "ENCSR000BLZ"
        assert entries[1]["accession"] == "ENCSR000AKS"

    def test_add_dedup_moves_to_top(self, history: SelectionHistory) -> None:
        """Re-adding an existing accession moves it to the top without duplicating."""
        history.add("ENCSR000AKS", SAMPLE_DATASET)
        history.add("ENCSR000BLZ", SAMPLE_DATASET_2)
        history.add("ENCSR000AKS", SAMPLE_DATASET)
        entries = history.get_entries()
        assert len(entries) == 2
        assert entries[0]["accession"] == "ENCSR000AKS"
        assert entries[1]["accession"] == "ENCSR000BLZ"

    def test_add_enforces_max_entries(self, history: SelectionHistory) -> None:
        """Adding beyond MAX_ENTRIES trims the oldest entries."""
        for i in range(35):
            history.add(f"ENCSR{i:06d}", {"accession": f"ENCSR{i:06d}"})
        entries = history.get_entries()
        assert len(entries) == SelectionHistory.MAX_ENTRIES
        # Most recent is first
        assert entries[0]["accession"] == "ENCSR000034"

    def test_add_truncates_long_description(self, history: SelectionHistory) -> None:
        """Descriptions longer than 120 chars are truncated."""
        long_desc = "A" * 200
        dataset = {**SAMPLE_DATASET, "description": long_desc}
        history.add("ENCSR000AKS", dataset)
        entries = history.get_entries()
        assert len(entries[0]["description"]) == 120
        assert entries[0]["description"].endswith("...")

    def test_add_short_description_unchanged(self, history: SelectionHistory) -> None:
        """Descriptions under 120 chars are stored as-is."""
        history.add("ENCSR000AKS", SAMPLE_DATASET)
        entries = history.get_entries()
        assert entries[0]["description"] == SAMPLE_DATASET["description"]

    def test_add_includes_timestamp(self, history: SelectionHistory) -> None:
        """Each entry has a timestamp field."""
        history.add("ENCSR000AKS", SAMPLE_DATASET)
        entries = history.get_entries()
        assert "timestamp" in entries[0]
        assert isinstance(entries[0]["timestamp"], str)

    def test_add_handles_missing_fields(self, history: SelectionHistory) -> None:
        """Adding a dataset with missing metadata fields uses empty strings."""
        minimal = {"accession": "ENCSR999ZZZ"}
        history.add("ENCSR999ZZZ", minimal)
        entries = history.get_entries()
        assert entries[0]["assay_term_name"] == ""
        assert entries[0]["biosample_term_name"] == ""
        assert entries[0]["organism"] == ""


class TestSelectionHistoryClear:
    """Tests for clear method."""

    def test_clear_empties_entries(self, history: SelectionHistory) -> None:
        """Clear removes all entries."""
        history.add("ENCSR000AKS", SAMPLE_DATASET)
        history.add("ENCSR000BLZ", SAMPLE_DATASET_2)
        history.clear()
        assert history.get_entries() == []

    def test_clear_deletes_file(
        self, history: SelectionHistory, history_path: Path
    ) -> None:
        """Clear removes the JSON file from disk."""
        history.add("ENCSR000AKS", SAMPLE_DATASET)
        assert history_path.exists()
        history.clear()
        assert not history_path.exists()

    def test_clear_noop_when_empty(
        self, history: SelectionHistory, history_path: Path
    ) -> None:
        """Clear on empty history does not error."""
        history.clear()
        assert history.get_entries() == []


class TestFormatEntryLabel:
    """Tests for format_entry_label static method."""

    def test_full_entry_label(self) -> None:
        """Label includes accession, assay, biosample, and organism."""
        entry = {
            "accession": "ENCSR000AKS",
            "assay_term_name": "ChIP-seq",
            "biosample_term_name": "K562",
            "organism": "Homo sapiens",
        }
        label = SelectionHistory.format_entry_label(entry)
        assert label == "ENCSR000AKS - ChIP-seq | K562 | Homo sapiens"

    def test_label_missing_biosample(self) -> None:
        """Label omits missing biosample gracefully."""
        entry = {
            "accession": "ENCSR000AKS",
            "assay_term_name": "ChIP-seq",
            "biosample_term_name": "",
            "organism": "Homo sapiens",
        }
        label = SelectionHistory.format_entry_label(entry)
        assert label == "ENCSR000AKS - ChIP-seq | Homo sapiens"

    def test_label_accession_only(self) -> None:
        """Label with only accession shows just the accession."""
        entry = {
            "accession": "ENCSR000AKS",
            "assay_term_name": "",
            "biosample_term_name": "",
            "organism": "",
        }
        label = SelectionHistory.format_entry_label(entry)
        assert label == "ENCSR000AKS"

    def test_label_missing_accession(self) -> None:
        """Label with missing accession falls back to 'unknown'."""
        entry = {}
        label = SelectionHistory.format_entry_label(entry)
        assert label == "unknown"

    def test_label_partial_metadata(self) -> None:
        """Label with only organism shows accession + organism."""
        entry = {
            "accession": "ENCSR000AKS",
            "assay_term_name": "",
            "biosample_term_name": "",
            "organism": "Mus musculus",
        }
        label = SelectionHistory.format_entry_label(entry)
        assert label == "ENCSR000AKS - Mus musculus"
