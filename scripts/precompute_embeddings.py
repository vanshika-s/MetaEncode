#!/usr/bin/env python3
# scripts/precompute_embeddings.py
"""Precompute embeddings for ENCODE experiments.

This script fetches experiments from the ENCODE API, processes metadata,
generates text embeddings, combines features, and caches everything for
fast application startup.

Usage:
    pip install -e .  # Install package first (one time)
    python scripts/precompute_embeddings.py --limit 1000
    python scripts/precompute_embeddings.py --limit all --batch-size 64
    python scripts/precompute_embeddings.py --refresh  # Force refresh
"""

import argparse
import sys
import time

import numpy as np

from src.api.encode_client import EncodeClient
from src.ml.embeddings import EmbeddingGenerator
from src.ml.feature_combiner import FeatureCombiner
from src.processing.metadata import MetadataProcessor
from src.utils.cache import CacheManager


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Precompute embeddings for ENCODE experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Precompute 100 experiments (quick test)
    python scripts/precompute_embeddings.py --limit 100

    # Precompute 1000 experiments
    python scripts/precompute_embeddings.py --limit 1000

    # Precompute all experiments (may take a while)
    python scripts/precompute_embeddings.py --limit all

    # Force refresh even if cache exists
    python scripts/precompute_embeddings.py --limit 500 --refresh
        """,
    )
    parser.add_argument(
        "--limit",
        type=str,
        default="100",
        help="Number of experiments to fetch ('all' for full dataset, default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation (default: 64)",
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
        help="Force refresh even if cached data exists",
    )
    return parser.parse_args()


def generate_embeddings_batched(
    texts: list[str],
    embedder: EmbeddingGenerator,
    batch_size: int = 64,
) -> np.ndarray:
    """Generate embeddings in batches to manage memory.

    Args:
        texts: List of text strings to embed.
        embedder: EmbeddingGenerator instance.
        batch_size: Number of texts per batch.

    Returns:
        NumPy array of embeddings, shape (n_texts, embedding_dim).
    """
    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_num = i // batch_size + 1
        batch = texts[i : i + batch_size]
        print(f"  Embedding batch {batch_num}/{n_batches} ({len(batch)} texts)...")
        embeddings = embedder.encode(batch, show_progress=False)
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)


def main() -> int:
    """Main entry point for precomputation.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args()
    start_time = time.time()

    print("=" * 60)
    print("MetaENCODE Precomputation Pipeline")
    print("=" * 60)

    # Initialize components
    print("\n[1/6] Initializing components...")
    cache_mgr = CacheManager(cache_dir=args.cache_dir)
    client = EncodeClient()
    processor = MetadataProcessor()
    embedder = EmbeddingGenerator()
    combiner = FeatureCombiner()

    # Check if cache exists
    if not args.refresh and cache_mgr.exists("combined_vectors"):
        print("\nCached data already exists. Use --refresh to force recomputation.")
        return 0

    # Parse limit (0 means "all" per EncodeClient.fetch_experiments API)
    limit = 0 if args.limit.lower() == "all" else int(args.limit)
    limit_str = "all" if limit == 0 else str(limit)

    # Fetch experiments
    print(f"\n[2/6] Fetching experiments from ENCODE API (limit={limit_str})...")
    try:
        raw_df = client.fetch_experiments(limit=limit)
        print(f"  Fetched {len(raw_df)} experiments")
    except Exception as e:
        print(f"  ERROR: Failed to fetch experiments: {e}")
        return 1

    if raw_df.empty:
        print("  ERROR: No experiments found")
        return 1

    # Process metadata
    print("\n[3/6] Processing metadata...")
    try:
        processed_df = processor.process(raw_df)
        print(f"  Processed {len(processed_df)} experiments")
    except Exception as e:
        print(f"  ERROR: Failed to process metadata: {e}")
        return 1

    # Generate text embeddings
    print(f"\n[4/6] Generating text embeddings (batch_size={args.batch_size})...")
    try:
        texts = processed_df["combined_text"].tolist()
        text_embeddings = generate_embeddings_batched(
            texts, embedder, batch_size=args.batch_size
        )
        print(f"  Generated embeddings: shape {text_embeddings.shape}")
    except Exception as e:
        print(f"  ERROR: Failed to generate embeddings: {e}")
        return 1

    # Fit feature combiner and generate combined vectors
    print("\n[5/6] Combining features (text + categorical + numeric)...")
    try:
        combiner.fit(processed_df, text_embedding_dim=text_embeddings.shape[1])
        combined_vectors = combiner.transform(processed_df, text_embeddings)

        breakdown = combiner.get_feature_breakdown()
        print(f"  Combined vector dimension: {combiner.feature_dim}")
        print("  Feature breakdown:")
        for name, dim in breakdown.items():
            print(f"    - {name}: {dim}")
    except Exception as e:
        print(f"  ERROR: Failed to combine features: {e}")
        return 1

    # Cache everything
    print("\n[6/6] Caching precomputed data...")
    try:
        cache_mgr.save("metadata", processed_df)
        print(f"  Saved metadata ({len(processed_df)} rows)")

        cache_mgr.save("embeddings", text_embeddings)
        print(f"  Saved text embeddings {text_embeddings.shape}")

        cache_mgr.save("combined_vectors", combined_vectors)
        print(f"  Saved combined vectors {combined_vectors.shape}")

        cache_mgr.save("feature_combiner", combiner)
        print("  Saved feature combiner")
    except Exception as e:
        print(f"  ERROR: Failed to save cache: {e}")
        return 1

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Precomputation Complete!")
    print("=" * 60)
    print(f"  Experiments: {len(processed_df)}")
    print(f"  Text embedding dim: {text_embeddings.shape[1]}")
    print(f"  Combined vector dim: {combiner.feature_dim}")
    print(f"  Cache directory: {args.cache_dir}")
    print(f"  Total time: {elapsed:.1f}s")
    print("\nYou can now run the Streamlit app:")
    print("  streamlit run app.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
