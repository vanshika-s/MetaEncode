# tests/test_utils/test_cache.py
"""Tests for CacheManager class."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.cache import CacheManager


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def cache_manager(temp_cache_dir: Path) -> CacheManager:
    """Create a CacheManager with temporary directory."""
    return CacheManager(cache_dir=str(temp_cache_dir))


class TestCacheManagerInit:
    """Tests for CacheManager initialization."""

    def test_init_creates_cache_dir(self, tmp_path: Path) -> None:
        """Test that init creates cache directory if it doesn't exist."""
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()

        CacheManager(cache_dir=str(cache_dir))

        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_init_with_default_dir(self) -> None:
        """Test initialization with default cache directory."""
        cache_mgr = CacheManager()
        assert cache_mgr.cache_dir == CacheManager.DEFAULT_CACHE_DIR

    def test_init_with_expiry(self, temp_cache_dir: Path) -> None:
        """Test initialization with expiry hours."""
        cache_mgr = CacheManager(cache_dir=str(temp_cache_dir), expiry_hours=24)
        assert cache_mgr.expiry_hours == 24


class TestCacheManagerSaveLoad:
    """Tests for save and load operations."""

    def test_save_and_load_dict(self, cache_manager: CacheManager) -> None:
        """Test saving and loading a dictionary."""
        data = {"key1": "value1", "key2": 42}
        cache_manager.save("test_dict", data)
        loaded = cache_manager.load("test_dict")
        assert loaded == data

    def test_save_and_load_numpy_array(self, cache_manager: CacheManager) -> None:
        """Test saving and loading a numpy array."""
        data = np.random.randn(100, 384).astype(np.float32)
        cache_manager.save("test_array", data)
        loaded = cache_manager.load("test_array")
        np.testing.assert_array_equal(loaded, data)

    def test_save_and_load_dataframe(self, cache_manager: CacheManager) -> None:
        """Test saving and loading a pandas DataFrame."""
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        cache_manager.save("test_df", data)
        loaded = cache_manager.load("test_df")
        pd.testing.assert_frame_equal(loaded, data)

    def test_save_returns_path(
        self, cache_manager: CacheManager, temp_cache_dir: Path
    ) -> None:
        """Test that save returns the cache file path."""
        path = cache_manager.save("test_key", {"data": 1})
        assert path == temp_cache_dir / "test_key.pkl"
        assert path.exists()

    def test_load_nonexistent_returns_none(self, cache_manager: CacheManager) -> None:
        """Test that loading nonexistent key returns None."""
        result = cache_manager.load("nonexistent_key")
        assert result is None

    def test_atomic_write_pattern(
        self, cache_manager: CacheManager, temp_cache_dir: Path
    ) -> None:
        """Test that save uses atomic write (temp file then rename)."""
        cache_manager.save("test_atomic", {"data": 1})

        # Check that no .tmp files remain
        tmp_files = list(temp_cache_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

        # Check that .pkl file exists
        pkl_files = list(temp_cache_dir.glob("*.pkl"))
        assert len(pkl_files) == 1


class TestCacheManagerExists:
    """Tests for exists method."""

    def test_exists_returns_true_for_saved_key(
        self, cache_manager: CacheManager
    ) -> None:
        """Test exists returns True for saved data."""
        cache_manager.save("test_key", "test_data")
        assert cache_manager.exists("test_key") is True

    def test_exists_returns_false_for_missing_key(
        self, cache_manager: CacheManager
    ) -> None:
        """Test exists returns False for missing key."""
        assert cache_manager.exists("nonexistent_key") is False


class TestCacheManagerDelete:
    """Tests for delete method."""

    def test_delete_removes_entry(self, cache_manager: CacheManager) -> None:
        """Test that delete removes cache entry."""
        cache_manager.save("test_key", "test_data")
        assert cache_manager.exists("test_key") is True

        result = cache_manager.delete("test_key")

        assert result is True
        assert cache_manager.exists("test_key") is False

    def test_delete_nonexistent_returns_false(
        self, cache_manager: CacheManager
    ) -> None:
        """Test that deleting nonexistent key returns False."""
        result = cache_manager.delete("nonexistent_key")
        assert result is False


class TestCacheManagerClear:
    """Tests for clear method."""

    def test_clear_removes_all_entries(self, cache_manager: CacheManager) -> None:
        """Test that clear removes all cache entries."""
        cache_manager.save("key1", "data1")
        cache_manager.save("key2", "data2")
        cache_manager.save("key3", "data3")

        count = cache_manager.clear()

        assert count == 3
        assert cache_manager.exists("key1") is False
        assert cache_manager.exists("key2") is False
        assert cache_manager.exists("key3") is False

    def test_clear_empty_cache_returns_zero(self, cache_manager: CacheManager) -> None:
        """Test that clearing empty cache returns 0."""
        count = cache_manager.clear()
        assert count == 0


class TestCacheManagerExpiration:
    """Tests for cache expiration."""

    def test_expired_entry_not_loaded(self, temp_cache_dir: Path) -> None:
        """Test that expired entries are not loaded."""
        # Create cache with very short expiry
        cache_mgr = CacheManager(cache_dir=str(temp_cache_dir), expiry_hours=0)

        cache_mgr.save("test_key", "test_data")

        # Entry should be considered expired immediately (expiry_hours=0)
        result = cache_mgr.load("test_key")
        assert result is None

    def test_expired_entry_deleted_on_load(self, temp_cache_dir: Path) -> None:
        """Test that expired entries are deleted when loading."""
        cache_mgr = CacheManager(cache_dir=str(temp_cache_dir), expiry_hours=0)

        cache_path = cache_mgr.save("test_key", "test_data")
        assert cache_path.exists()

        cache_mgr.load("test_key")

        # File should be deleted
        assert not cache_path.exists()

    def test_no_expiry_by_default(self, cache_manager: CacheManager) -> None:
        """Test that default cache has no expiration."""
        assert cache_manager.expiry_hours is None

        cache_manager.save("test_key", "test_data")
        # Should still be loadable
        result = cache_manager.load("test_key")
        assert result == "test_data"


class TestCacheManagerCorruption:
    """Tests for handling corrupted cache files."""

    def test_corrupted_file_returns_none(
        self, cache_manager: CacheManager, temp_cache_dir: Path
    ) -> None:
        """Test that corrupted cache files return None."""
        # Create a corrupted pickle file
        cache_path = temp_cache_dir / "corrupted.pkl"
        with open(cache_path, "wb") as f:
            f.write(b"this is not a valid pickle file")

        result = cache_manager.load("corrupted")
        assert result is None

    def test_corrupted_file_deleted(
        self, cache_manager: CacheManager, temp_cache_dir: Path
    ) -> None:
        """Test that corrupted cache files are deleted."""
        cache_path = temp_cache_dir / "corrupted.pkl"
        with open(cache_path, "wb") as f:
            f.write(b"invalid pickle data")

        assert cache_path.exists()
        cache_manager.load("corrupted")
        assert not cache_path.exists()

    def test_truncated_file_handled(
        self, cache_manager: CacheManager, temp_cache_dir: Path
    ) -> None:
        """Test that truncated pickle files are handled gracefully."""
        # Create a valid pickle, then truncate it
        cache_manager.save("test_key", {"key": "value"})
        cache_path = temp_cache_dir / "test_key.pkl"

        # Truncate the file
        with open(cache_path, "r+b") as f:
            f.truncate(10)

        result = cache_manager.load("test_key")
        assert result is None


class TestCacheManagerOverwrite:
    """Tests for overwriting existing cache entries."""

    def test_save_overwrites_existing(self, cache_manager: CacheManager) -> None:
        """Test that saving to existing key overwrites."""
        cache_manager.save("test_key", "original_data")
        cache_manager.save("test_key", "new_data")

        result = cache_manager.load("test_key")
        assert result == "new_data"

    def test_save_large_data(self, cache_manager: CacheManager) -> None:
        """Test saving large data."""
        # 100MB of random data
        large_data = np.random.randn(10000, 1000).astype(np.float32)
        cache_manager.save("large_data", large_data)
        loaded = cache_manager.load("large_data")
        np.testing.assert_array_equal(loaded, large_data)


# ============================================================================
# Additional Edge Case Tests for Coverage
# ============================================================================


class TestCacheManagerCoverageEdgeCases:
    """Additional edge case tests for CacheManager coverage."""

    def test_exists_returns_false_when_expired(self, temp_cache_dir: Path) -> None:
        """Line 110: exists() returns False for expired entry."""
        # Create cache with expiry_hours=0 (any file is immediately expired)
        cache_mgr = CacheManager(cache_dir=str(temp_cache_dir), expiry_hours=0)
        cache_mgr.save("test_key", "test_data")

        # exists() should return False because expiry_hours=0 means
        # everything is expired
        assert cache_mgr.exists("test_key") is False

    def test_is_expired_missing_file(self, temp_cache_dir: Path) -> None:
        """Line 167: _is_expired() returns True for missing file."""
        cache_mgr = CacheManager(cache_dir=str(temp_cache_dir), expiry_hours=24)
        nonexistent_path = temp_cache_dir / "nonexistent.pkl"

        # _is_expired should return True for missing file
        result = cache_mgr._is_expired(nonexistent_path)
        assert result is True
