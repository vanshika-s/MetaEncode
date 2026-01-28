# src/utils/cache.py
"""Caching utilities for embeddings and API responses.

This module provides caching functionality to avoid recomputing embeddings
and re-fetching data from the ENCODE API. Supports file-based caching
with optional expiration.
"""

import pickle
import time
from pathlib import Path
from typing import Any, Optional


class CacheManager:
    """Manage caching of embeddings and API responses.

    Provides file-based caching with optional expiration. Useful for:
    - Caching precomputed embeddings
    - Caching API responses to reduce rate limit issues
    - Persisting session state across restarts

    Example:
        >>> cache = CacheManager(cache_dir="data/cache")
        >>> cache.save("embeddings", embeddings_array)
        >>> loaded = cache.load("embeddings")
    """

    DEFAULT_CACHE_DIR = Path("data/cache")

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        expiry_hours: Optional[int] = None,
    ) -> None:
        """Initialize the cache manager.

        Args:
            cache_dir: Directory for cache files. Defaults to data/cache.
            expiry_hours: Hours before cache expires. None means no expiration.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        self.expiry_hours = expiry_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save(self, key: str, data: Any) -> Path:
        """Save data to cache.

        Uses atomic write pattern to prevent corruption from partial writes.

        Args:
            key: Unique identifier for the cached data.
            data: Data to cache (must be picklable).

        Returns:
            Path to the cache file.
        """
        cache_path = self._get_cache_path(key)
        tmp_path = cache_path.with_suffix(".tmp")

        with open(tmp_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Atomic rename (POSIX-compliant)
        tmp_path.rename(cache_path)
        return cache_path

    def load(self, key: str) -> Optional[Any]:
        """Load data from cache.

        Args:
            key: Unique identifier for the cached data.

        Returns:
            Cached data, or None if not found or expired.
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        if self._is_expired(cache_path):
            self.delete(key)
            return None

        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError) as e:
            # Corrupted cache file - delete and return None
            print(f"Warning: Corrupted cache file {cache_path}, deleting: {e}")
            self.delete(key)
            return None

    def exists(self, key: str) -> bool:
        """Check if cache entry exists and is valid.

        Args:
            key: Unique identifier for the cached data.

        Returns:
            True if cache entry exists and is not expired.
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return False

        if self._is_expired(cache_path):
            return False

        return True

    def delete(self, key: str) -> bool:
        """Delete a cache entry.

        Args:
            key: Unique identifier for the cached data.

        Returns:
            True if entry was deleted, False if it didn't exist.
        """
        cache_path = self._get_cache_path(key)

        if cache_path.exists():
            cache_path.unlink()
            return True

        return False

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries deleted.
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
            count += 1
        return count

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key.

        Args:
            key: Cache key.

        Returns:
            Path to the cache file.
        """
        return self.cache_dir / f"{key}.pkl"

    def _is_expired(self, cache_path: Path) -> bool:
        """Check if a cache file is expired.

        Args:
            cache_path: Path to the cache file.

        Returns:
            True if the file is expired, False otherwise.
        """
        if self.expiry_hours is None:
            return False

        if not cache_path.exists():
            return True

        mtime = cache_path.stat().st_mtime
        age_hours = (time.time() - mtime) / 3600

        return age_hours > self.expiry_hours
