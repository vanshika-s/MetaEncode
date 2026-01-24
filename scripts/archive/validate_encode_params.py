#!/usr/bin/env python3
"""Validate ENCODE API parameters by running actual queries.

RUN THIS SCRIPT LOCALLY (not in CI/restricted environments) to:
1. Verify parameters actually work against ENCODE API
2. Get result counts to order options by popularity
3. Identify invalid/zero-result options that should be removed

Usage:
    python scripts/validate_encode_params.py

Output is saved to scripts/encode_validation_results.json
"""

import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_URL = "https://www.encodeproject.org"


def query_encode(params: dict[str, Any], limit: int = 0) -> dict:
    """Query ENCODE API and return result count and sample accessions.

    Args:
        params: Query parameters
        limit: If 0, just get count. If > 0, also get sample results.

    Returns:
        Dict with 'count', 'url', and optionally 'samples'
    """
    query_params = {
        "type": "Experiment",
        "format": "json",
        **params,
    }
    if limit > 0:
        query_params["limit"] = limit
    else:
        query_params["limit"] = 0  # Just get count, no results

    url = f"{BASE_URL}/search/?{urlencode(query_params)}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        result = {
            "count": data.get("total", 0),
            "url": url,
        }

        if limit > 0:
            result["samples"] = [
                exp.get("accession", "N/A") for exp in data.get("@graph", [])[:5]
            ]

        return result
    except Exception as e:
        return {"count": -1, "error": str(e), "url": url}


def test_assay_types():
    """Test each assay type and get result counts."""
    print("\n" + "=" * 70)
    print("TESTING ASSAY TYPES (assay_term_name parameter)")
    print("=" * 70)

    from src.ui.vocabularies import ASSAY_TYPES

    results = []
    for assay_name in ASSAY_TYPES.keys():
        result = query_encode({"assay_term_name": assay_name})
        results.append(
            {
                "assay": assay_name,
                "count": result["count"],
                "error": result.get("error"),
            }
        )
        status = (
            f"{result['count']:6d}"
            if result["count"] >= 0
            else f"ERROR: {result.get('error', 'unknown')}"
        )
        print(f"  {assay_name:50s} -> {status}")
        time.sleep(0.2)  # Rate limiting

    # Sort by count descending
    results.sort(key=lambda x: x["count"], reverse=True)

    print("\n### SORTED BY POPULARITY ###")
    for r in results[:20]:  # Top 20
        if r["count"] > 0:
            print(f"  {r['count']:6d}  {r['assay']}")

    print("\n### ZERO RESULTS (potentially invalid) ###")
    for r in results:
        if r["count"] == 0:
            print(f"  {r['assay']}")

    return results


def test_organisms():
    """Test organism scientific names."""
    print("\n" + "=" * 70)
    print("TESTING ORGANISMS")
    print("=" * 70)

    organisms = [
        ("Homo sapiens", "human"),
        ("Mus musculus", "mouse"),
        ("Drosophila melanogaster", "fly"),
        ("Caenorhabditis elegans", "worm"),
    ]

    param_paths = [
        "replicates.library.biosample.donor.organism.scientific_name",
        "replicates.library.biosample.organism.scientific_name",
        "biosample_ontology.organism.scientific_name",
    ]

    for param_path in param_paths:
        print(f"\n### Testing parameter path: {param_path}")
        for sci_name, common_name in organisms:
            result = query_encode({param_path: sci_name})
            status = f"{result['count']:6d}" if result["count"] >= 0 else "ERROR"
            print(f"  {common_name:20s} ({sci_name:30s}) -> {status}")
            time.sleep(0.2)


def test_life_stage():
    """Test life_stage parameter paths."""
    print("\n" + "=" * 70)
    print("TESTING LIFE_STAGE PARAMETER PATHS")
    print("=" * 70)

    life_stages = ["embryonic", "postnatal", "adult", "unknown"]

    param_paths = [
        "replicates.library.biosample.life_stage",
        "biosample_ontology.life_stage",
        "life_stage",
    ]

    for param_path in param_paths:
        print(f"\n### Testing parameter path: {param_path}")
        for stage in life_stages:
            result = query_encode({param_path: stage})
            status = f"{result['count']:6d}" if result["count"] >= 0 else "ERROR"
            print(f"  {stage:20s} -> {status}")
            time.sleep(0.2)


def test_target():
    """Test target parameter paths."""
    print("\n" + "=" * 70)
    print("TESTING TARGET PARAMETER PATHS")
    print("=" * 70)

    targets = ["CTCF", "H3K27ac", "H3K4me3", "POLR2A"]

    param_paths = [
        "target.label",
        "target.name",
        "target",
    ]

    for param_path in param_paths:
        print(f"\n### Testing parameter path: {param_path}")
        for target in targets:
            result = query_encode({param_path: target})
            status = f"{result['count']:6d}" if result["count"] >= 0 else "ERROR"
            print(f"  {target:20s} -> {status}")
            time.sleep(0.2)


def test_biosamples():
    """Test biosample term names."""
    print("\n" + "=" * 70)
    print("TESTING BIOSAMPLES (biosample_ontology.term_name)")
    print("=" * 70)

    # Common cell lines and tissues
    biosamples = [
        "K562",
        "GM12878",
        "H1-hESC",
        "HepG2",
        "A549",
        "HeLa-S3",
        "MCF-7",
        "IMR-90",
        "HUVEC",
        "SK-N-SH",
        "cerebellum",
        "liver",
        "heart",
        "kidney",
        "lung",
        "brain",
        "spleen",
        "thymus",
        "whole blood",
    ]

    results = []
    for biosample in biosamples:
        result = query_encode({"biosample_ontology.term_name": biosample})
        results.append(
            {
                "biosample": biosample,
                "count": result["count"],
            }
        )
        status = f"{result['count']:6d}" if result["count"] >= 0 else "ERROR"
        print(f"  {biosample:30s} -> {status}")
        time.sleep(0.2)

    # Sort by count
    results.sort(key=lambda x: x["count"], reverse=True)

    print("\n### SORTED BY POPULARITY ###")
    for r in results:
        if r["count"] > 0:
            print(f"  {r['count']:6d}  {r['biosample']}")

    return results


def test_combined_query():
    """Test a combined query to verify all parameters work together."""
    print("\n" + "=" * 70)
    print("TESTING COMBINED QUERIES")
    print("=" * 70)

    # Test query that should return results
    test_queries = [
        {
            "name": "ChIP-seq + mouse + cerebellum",
            "params": {
                "assay_term_name": "ChIP-seq",
                "replicates.library.biosample.donor.organism.scientific_name": "Mus musculus",
                "biosample_ontology.term_name": "cerebellum",
            },
        },
        {
            "name": "ChIP-seq + mouse + CTCF",
            "params": {
                "assay_term_name": "ChIP-seq",
                "replicates.library.biosample.donor.organism.scientific_name": "Mus musculus",
                "target.label": "CTCF",
            },
        },
        {
            "name": "RNA-seq + human + K562",
            "params": {
                "assay_term_name": "RNA-seq",
                "replicates.library.biosample.donor.organism.scientific_name": "Homo sapiens",
                "biosample_ontology.term_name": "K562",
            },
        },
        {
            "name": "ATAC-seq + mouse + adult",
            "params": {
                "assay_term_name": "ATAC-seq",
                "replicates.library.biosample.donor.organism.scientific_name": "Mus musculus",
                "replicates.library.biosample.life_stage": "adult",
            },
        },
    ]

    for query in test_queries:
        result = query_encode(query["params"], limit=5)
        print(f"\n{query['name']}:")
        print(f"  Count: {result['count']}")
        print(f"  URL: {result['url']}")
        if result.get("samples"):
            print(f"  Samples: {', '.join(result['samples'])}")
        if result.get("error"):
            print(f"  ERROR: {result['error']}")
        time.sleep(0.3)


def save_results(results: dict, filename: str):
    """Save results to JSON file."""
    output_path = Path(__file__).parent / filename
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    print("=" * 70)
    print("ENCODE API PARAMETER VALIDATION")
    print("=" * 70)
    print("\nThis script queries the ENCODE API to verify parameters work correctly.")
    print("It will test assay types, organisms, biosamples, life stages, and targets.")

    all_results = {}

    # Test organisms first (fastest)
    test_organisms()

    # Test life_stage parameter paths
    test_life_stage()

    # Test target parameter paths
    test_target()

    # Test biosamples
    biosample_results = test_biosamples()
    all_results["biosamples"] = biosample_results

    # Test assay types (takes longer)
    assay_results = test_assay_types()
    all_results["assays"] = assay_results

    # Test combined queries
    test_combined_query()

    # Save results
    save_results(all_results, "encode_validation_results.json")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
