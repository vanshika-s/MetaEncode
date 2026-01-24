#!/usr/bin/env python3
"""Explore ENCODE assay types via the API.

This script queries the ENCODE API to discover what assay types actually exist
and their frequency in the database.
"""

import sys
from collections import Counter
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.encode_client import EncodeClient


def main():
    client = EncodeClient()

    print("=" * 60)
    print("Fetching experiments from ENCODE API...")
    print("=" * 60)

    # Fetch a large sample of experiments to see what assay types exist
    df = client.fetch_experiments(limit=500)

    print(f"\nFetched {len(df)} experiments\n")

    # Count assay types
    print("=" * 60)
    print("ASSAY TYPES (sorted by frequency)")
    print("=" * 60)
    assay_counts = Counter(df["assay_term_name"])
    for assay, count in assay_counts.most_common():
        print(f"  {count:4d}  {assay}")

    print(f"\nTotal unique assay types: {len(assay_counts)}")

    # Check if our vocabularies match
    from src.ui.vocabularies import ASSAY_TYPES

    print("\n" + "=" * 60)
    print("VALIDATION: Checking vocabulary against actual ENCODE data")
    print("=" * 60)

    actual_assays = set(assay_counts.keys())
    vocab_assays = set(ASSAY_TYPES.keys())

    # Assays in ENCODE but not in our vocabulary
    missing_from_vocab = actual_assays - vocab_assays
    if missing_from_vocab:
        print(
            f"\nAssays in ENCODE but NOT in our vocabulary ({len(missing_from_vocab)}):"
        )
        for assay in sorted(missing_from_vocab):
            print(f"  - {assay}  (count: {assay_counts[assay]})")

    # Assays in vocabulary but not found in ENCODE sample
    not_found_in_encode = vocab_assays - actual_assays
    if not_found_in_encode:
        print(
            f"\nAssays in vocabulary but NOT found in sample ({len(not_found_in_encode)}):"
        )
        for assay in sorted(not_found_in_encode):
            print(f"  - {assay}")

    # Test specific assays that failed
    print("\n" + "=" * 60)
    print("TESTING SPECIFIC ASSAY SEARCHES")
    print("=" * 60)

    test_assays = ["4C", "CUT&RUN", "ChIP-seq", "RNA-seq", "Hi-C", "ATAC-seq"]
    for assay in test_assays:
        try:
            results = client.fetch_experiments(assay_type=assay, limit=5)
            print(f"  {assay}: {len(results)} results")
        except Exception as e:
            print(f"  {assay}: ERROR - {e}")


if __name__ == "__main__":
    main()
