#!/usr/bin/env python3
"""Explore cached ENCODE data from pickle files.

This script examines the cached pickle files to see what data is available.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.cache import CacheManager


def main():
    cache_mgr = CacheManager()

    print("=" * 60)
    print("Examining cached ENCODE data")
    print("=" * 60)

    # Check what's cached
    for name in ["metadata", "embeddings", "combined_vectors", "feature_combiner"]:
        if cache_mgr.exists(name):
            print(f"\n{name}: EXISTS")
            try:
                data = cache_mgr.load(name)
                print(f"  Type: {type(data)}")
                if hasattr(data, "shape"):
                    print(f"  Shape: {data.shape}")
                if hasattr(data, "__len__"):
                    print(f"  Length: {len(data)}")
                if hasattr(data, "columns"):
                    print(f"  Columns: {list(data.columns)}")
                    if len(data) > 0:
                        print(f"  First row sample:")
                        for col in data.columns:
                            val = data.iloc[0][col]
                            if isinstance(val, str) and len(val) > 50:
                                val = val[:50] + "..."
                            print(f"    {col}: {val}")
            except Exception as e:
                print(f"  Error loading: {e}")
        else:
            print(f"\n{name}: NOT FOUND")

    # If metadata exists and has data, analyze it
    if cache_mgr.exists("metadata"):
        try:
            metadata = cache_mgr.load("metadata")
            if hasattr(metadata, "__len__") and len(metadata) > 0:
                from collections import Counter

                print("\n" + "=" * 60)
                print("ASSAY TYPES in cached data (sorted by frequency)")
                print("=" * 60)

                if "assay_term_name" in metadata.columns:
                    assay_counts = Counter(metadata["assay_term_name"])
                    for assay, count in assay_counts.most_common():
                        print(f"  {count:4d}  {assay}")
                    print(f"\nTotal unique assay types: {len(assay_counts)}")

                print("\n" + "=" * 60)
                print("ORGANISMS in cached data")
                print("=" * 60)

                if "organism" in metadata.columns:
                    org_counts = Counter(metadata["organism"])
                    for org, count in org_counts.most_common():
                        print(f"  {count:4d}  {org}")

                print("\n" + "=" * 60)
                print("BIOSAMPLES in cached data (top 30)")
                print("=" * 60)

                if "biosample_term_name" in metadata.columns:
                    biosample_counts = Counter(metadata["biosample_term_name"])
                    for biosample, count in biosample_counts.most_common(30):
                        print(f"  {count:4d}  {biosample}")
                    print(f"\nTotal unique biosamples: {len(biosample_counts)}")

                print("\n" + "=" * 60)
                print("SAMPLE DESCRIPTIONS (first 10)")
                print("=" * 60)

                if "description" in metadata.columns:
                    for i, desc in enumerate(metadata["description"].head(10)):
                        print(f"\n[{i+1}] {desc}")
        except Exception as e:
            print(f"Error analyzing metadata: {e}")


if __name__ == "__main__":
    main()
