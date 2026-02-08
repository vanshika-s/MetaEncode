#!/usr/bin/env python3
# scripts/precompute_visualizations.py
"""Precompute dimensionality reduction coordinates for global visualizations.

This script loads the cached embeddings, runs PCA, t-SNE, and UMAP
dimensionality reduction, and saves the 2D coordinates to data/cache/.
These precomputed coordinates are loaded instantly when a user requests
a global ("All Datasets") visualization, avoiding expensive on-the-fly
computation (especially for UMAP).

Usage:
    python scripts/precompute_visualizations.py
    python scripts/precompute_visualizations.py --methods pca umap
    python scripts/precompute_visualizations.py --refresh
    python scripts/precompute_visualizations.py --no-filter-outliers
"""

import argparse
import sys
import time

import numpy as np

from src.utils.cache import CacheManager
from src.visualization.plots import DimensionalityReducer, percentile_range_filtering

# Cache key prefix for precomputed visualization coordinates
VIZ_CACHE_PREFIX = "viz_coords"

# Methods to precompute
ALL_METHODS = ("pca", "t-sne", "umap")


def cache_key(method: str, filtered: bool) -> str:
    """Build the cache key for a given method and filter setting.

    Args:
        method: Dimensionality reduction method name.
        filtered: Whether outlier filtering was applied.

    Returns:
        Cache key string like 'viz_coords_pca_filtered'.
    """
    suffix = "filtered" if filtered else "unfiltered"
    return f"{VIZ_CACHE_PREFIX}_{method.replace('-', '_')}_{suffix}"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Precompute visualization coordinates for global views.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Precompute all methods (PCA, t-SNE, UMAP) with and without filtering
    python scripts/precompute_visualizations.py

    # Precompute only PCA and UMAP
    python scripts/precompute_visualizations.py --methods pca umap

    # Force refresh even if precomputed data exists
    python scripts/precompute_visualizations.py --refresh

    # Only precompute unfiltered variants
    python scripts/precompute_visualizations.py --no-filter-outliers
        """,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(ALL_METHODS),
        default=list(ALL_METHODS),
        help="Reduction methods to precompute (default: all)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Cache directory (default: data/cache)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh even if precomputed data exists",
    )
    parser.add_argument(
        "--no-filter-outliers",
        action="store_true",
        help="Skip precomputing filtered (outlier-removed) variants",
    )
    return parser.parse_args()


def precompute_method(
    method: str,
    embeddings: np.ndarray,
    filtered: bool,
    cache_mgr: CacheManager,
    refresh: bool,
) -> bool:
    """Precompute and cache 2D coordinates for one method/filter combination.

    Args:
        method: Dimensionality reduction method.
        embeddings: Input embedding array.
        filtered: Whether these are outlier-filtered embeddings.
        cache_mgr: Cache manager instance.
        refresh: Whether to overwrite existing cache.

    Returns:
        True if computation was performed, False if skipped.
    """
    key = cache_key(method, filtered)
    label = f"{method.upper()} ({'filtered' if filtered else 'unfiltered'})"

    if not refresh and cache_mgr.exists(key):
        print(f"  {label}: already cached, skipping (use --refresh to overwrite)")
        return False

    print(f"  {label}: computing...", end="", flush=True)
    start = time.time()

    reducer = DimensionalityReducer(method=method)
    coords_2d = reducer.fit_transform(embeddings)

    # Save coords and variance ratio (for PCA axis labels)
    data = {
        "coords_2d": coords_2d,
        "variance_ratio": reducer.variance_ratio_,
    }

    # For filtered variants, also save the boolean mask so metadata
    # can be filtered consistently at load time
    cache_mgr.save(key, data)

    elapsed = time.time() - start
    print(f" done ({elapsed:.1f}s, shape {coords_2d.shape})")
    return True


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args()
    start_time = time.time()

    print("=" * 60)
    print("MetaENCODE Visualization Precomputation")
    print("=" * 60)

    # Load cached embeddings
    print("\n[1/3] Loading cached embeddings...")
    cache_mgr = CacheManager(cache_dir=args.cache_dir)

    embeddings = cache_mgr.load("embeddings")
    metadata = cache_mgr.load("metadata")

    if embeddings is None or metadata is None:
        print("  ERROR: No cached embeddings or metadata found.")
        print("  Run scripts/precompute_embeddings.py first.")
        return 1

    print(f"  Loaded embeddings: {embeddings.shape}")
    print(f"  Loaded metadata: {metadata.shape[0]} experiments")

    # Prepare filtered embeddings (outlier removal)
    print("\n[2/3] Preparing embeddings...")
    filtered_embeddings, filter_mask = percentile_range_filtering(embeddings)
    print(f"  Unfiltered: {embeddings.shape[0]} samples")
    print(f"  Filtered (5-95th percentile): {filtered_embeddings.shape[0]} samples")

    # Save the filter mask so the frontend can reconstruct filtered metadata
    cache_mgr.save(f"{VIZ_CACHE_PREFIX}_filter_mask", filter_mask)

    # Precompute each method
    print("\n[3/3] Precomputing dimensionality reductions...")
    computed = 0

    for method in args.methods:
        # Unfiltered variant (always)
        if precompute_method(method, embeddings, False, cache_mgr, args.refresh):
            computed += 1

        # Filtered variant (unless --no-filter-outliers)
        if not args.no_filter_outliers:
            if precompute_method(
                method, filtered_embeddings, True, cache_mgr, args.refresh
            ):
                computed += 1

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Visualization Precomputation Complete!")
    print("=" * 60)
    print(f"  Methods: {', '.join(m.upper() for m in args.methods)}")
    print(f"  Computed: {computed} new projections")
    print(f"  Cache directory: {args.cache_dir}")
    print(f"  Total time: {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())