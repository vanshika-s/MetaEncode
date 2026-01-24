#!/usr/bin/env python3
"""Validate and order vocabularies based on actual ENCODE data.

RUN THIS SCRIPT LOCALLY to:
1. Query ENCODE API for each vocabulary option
2. Record result counts
3. Generate updated vocabularies ordered by popularity
4. Identify options that return zero results (should be removed)

Usage:
    python scripts/validate_and_order_vocabularies.py

Output:
    - scripts/encode_validation_results.json (raw results)
    - scripts/ordered_vocabularies.py (generated code to copy)
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
RATE_LIMIT_DELAY = 0.15  # seconds between requests


def query_encode_count(params: dict[str, Any]) -> tuple[int, str]:
    """Query ENCODE API and return result count.

    Returns:
        Tuple of (count, error_message). Count is -1 on error.
    """
    query_params = {
        "type": "Experiment",
        "format": "json",
        "limit": 0,  # Just get count
        **params,
    }
    url = f"{BASE_URL}/search/?{urlencode(query_params)}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("total", 0), ""
    except Exception as e:
        return -1, str(e)


def test_parameter_path(param_path: str, test_values: list[str]) -> dict:
    """Test if a parameter path works with given values."""
    print(f"\n  Testing: {param_path}")
    results = {}
    working = False

    for value in test_values:
        count, error = query_encode_count({param_path: value})
        results[value] = count
        if count > 0:
            working = True
        status = f"{count:6d}" if count >= 0 else f"ERROR"
        print(f"    {value:30s} -> {status}")
        time.sleep(RATE_LIMIT_DELAY)

    return {"path": param_path, "working": working, "results": results}


def find_working_parameter_path(
    param_name: str, paths: list[str], test_values: list[str]
) -> str:
    """Find which parameter path works for a given parameter type."""
    print(f"\n{'='*60}")
    print(f"FINDING WORKING PATH FOR: {param_name}")
    print(f"{'='*60}")

    for path in paths:
        result = test_parameter_path(path, test_values[:2])  # Test with first 2 values
        if result["working"]:
            print(f"\n  ✓ WORKING PATH: {path}")
            return path

    print(f"\n  ✗ NO WORKING PATH FOUND")
    return ""


def get_assay_counts() -> list[dict]:
    """Get result counts for all assay types."""
    print(f"\n{'='*60}")
    print("TESTING ASSAY TYPES")
    print(f"{'='*60}")

    from src.ui.vocabularies import ASSAY_TYPES

    results = []
    for assay_name, display_name in ASSAY_TYPES.items():
        count, error = query_encode_count({"assay_term_name": assay_name})
        results.append(
            {
                "key": assay_name,
                "display": display_name,
                "count": count,
                "error": error,
            }
        )
        status = f"{count:6d}" if count >= 0 else f"ERROR: {error[:30]}"
        print(f"  {assay_name:50s} -> {status}")
        time.sleep(RATE_LIMIT_DELAY)

    return results


def get_biosample_counts(organism_path: str) -> list[dict]:
    """Get result counts for common biosamples."""
    print(f"\n{'='*60}")
    print("TESTING BIOSAMPLES")
    print(f"{'='*60}")

    # Common cell lines and tissues to test
    biosamples = [
        # ENCODE Tier 1
        "K562",
        "GM12878",
        "H1-hESC",
        # ENCODE Tier 2
        "A549",
        "HeLa-S3",
        "HepG2",
        "HUVEC",
        "IMR-90",
        "MCF-7",
        "SK-N-SH",
        # Common tissues
        "liver",
        "heart",
        "kidney",
        "lung",
        "brain",
        "spleen",
        "thymus",
        "cerebellum",
        "hippocampus",
        "cortex",
        "whole blood",
        # Other cell lines
        "HEK293",
        "Jurkat",
        "NHEK",
        "Caco-2",
        "U2OS",
        "PC-3",
        "LNCaP",
    ]

    results = []
    for biosample in biosamples:
        count, error = query_encode_count({"biosample_ontology.term_name": biosample})
        results.append(
            {
                "key": biosample,
                "count": count,
                "error": error,
            }
        )
        status = f"{count:6d}" if count >= 0 else f"ERROR"
        print(f"  {biosample:30s} -> {status}")
        time.sleep(RATE_LIMIT_DELAY)

    return results


def get_target_counts() -> list[dict]:
    """Get result counts for common ChIP-seq targets."""
    print(f"\n{'='*60}")
    print("TESTING TARGETS (for ChIP-seq)")
    print(f"{'='*60}")

    targets = [
        "H3K27ac",
        "H3K4me3",
        "H3K4me1",
        "H3K27me3",
        "H3K9me3",
        "H3K36me3",
        "H3K9ac",
        "H4K20me1",
        "H3K79me2",
        "CTCF",
        "POLR2A",
        "EP300",
        "RAD21",
        "SMC3",
    ]

    # First find working path
    paths = ["target.label", "target.name", "target.investigated_as"]
    working_path = ""

    for path in paths:
        count, _ = query_encode_count({path: "CTCF"})
        if count > 0:
            working_path = path
            print(f"  Using parameter path: {path}")
            break

    if not working_path:
        print("  ERROR: No working target parameter path found")
        return []

    results = []
    for target in targets:
        count, error = query_encode_count({working_path: target})
        results.append(
            {
                "key": target,
                "count": count,
                "error": error,
                "path": working_path,
            }
        )
        status = f"{count:6d}" if count >= 0 else f"ERROR"
        print(f"  {target:20s} -> {status}")
        time.sleep(RATE_LIMIT_DELAY)

    return results


def get_life_stage_counts() -> list[dict]:
    """Get result counts for life stages."""
    print(f"\n{'='*60}")
    print("TESTING LIFE STAGES")
    print(f"{'='*60}")

    stages = ["embryonic", "postnatal", "adult", "unknown", "child", "newborn"]

    # Find working path
    paths = [
        "replicates.library.biosample.life_stage",
        "biosample_ontology.classification",
    ]

    working_path = ""
    for path in paths:
        count, _ = query_encode_count({path: "adult"})
        if count > 0:
            working_path = path
            print(f"  Using parameter path: {path}")
            break

    if not working_path:
        print("  WARNING: No working life_stage path found - will skip")
        return []

    results = []
    for stage in stages:
        count, error = query_encode_count({working_path: stage})
        results.append(
            {
                "key": stage,
                "count": count,
                "error": error,
                "path": working_path,
            }
        )
        status = f"{count:6d}" if count >= 0 else f"ERROR"
        print(f"  {stage:20s} -> {status}")
        time.sleep(RATE_LIMIT_DELAY)

    return results


def generate_ordered_code(results: dict) -> str:
    """Generate Python code with vocabularies ordered by popularity."""
    code = """# Generated by validate_and_order_vocabularies.py
# Vocabularies ordered by ENCODE result count (most popular first)

# ASSAY_TYPES ordered by popularity
ASSAY_TYPES_ORDERED = {
"""

    # Sort assays by count
    assays = sorted(results.get("assays", []), key=lambda x: x["count"], reverse=True)
    for a in assays:
        if a["count"] > 0:
            code += f'    "{a["key"]}": "{a["display"]}",  # {a["count"]} results\n'

    code += "}\n\n# Zero-result assays (consider removing):\n# "
    zero_assays = [a["key"] for a in assays if a["count"] == 0]
    code += ", ".join(zero_assays) if zero_assays else "None"

    code += "\n\n# BIOSAMPLES ordered by popularity\nBIOSAMPLES_ORDERED = [\n"
    biosamples = sorted(
        results.get("biosamples", []), key=lambda x: x["count"], reverse=True
    )
    for b in biosamples:
        if b["count"] > 0:
            code += f'    "{b["key"]}",  # {b["count"]} results\n'
    code += "]\n"

    code += "\n# TARGETS ordered by popularity\nTARGETS_ORDERED = [\n"
    targets = sorted(results.get("targets", []), key=lambda x: x["count"], reverse=True)
    for t in targets:
        if t["count"] > 0:
            code += f'    "{t["key"]}",  # {t["count"]} results\n'
    code += "]\n"

    # Add working parameter paths
    code += "\n# Working ENCODE API parameter paths:\n"
    if results.get("working_paths"):
        for name, path in results["working_paths"].items():
            code += f"# {name}: {path}\n"

    return code


def main():
    print("=" * 60)
    print("ENCODE VOCABULARY VALIDATION AND ORDERING")
    print("=" * 60)
    print("\nThis queries the ENCODE API to validate vocabulary options")
    print("and determine their popularity for proper ordering.\n")

    all_results = {
        "working_paths": {},
    }

    # 1. Find working organism path
    org_paths = [
        "replicates.library.biosample.donor.organism.scientific_name",
        "replicates.library.biosample.organism.scientific_name",
    ]
    org_path = find_working_parameter_path(
        "organism", org_paths, ["Homo sapiens", "Mus musculus"]
    )
    all_results["working_paths"]["organism"] = org_path

    # 2. Get life stage info
    life_results = get_life_stage_counts()
    all_results["life_stages"] = life_results
    if life_results and life_results[0].get("path"):
        all_results["working_paths"]["life_stage"] = life_results[0]["path"]

    # 3. Get target info
    target_results = get_target_counts()
    all_results["targets"] = target_results
    if target_results and target_results[0].get("path"):
        all_results["working_paths"]["target"] = target_results[0]["path"]

    # 4. Get biosample counts
    biosample_results = get_biosample_counts(org_path)
    all_results["biosamples"] = biosample_results

    # 5. Get assay counts (this takes longest)
    assay_results = get_assay_counts()
    all_results["assays"] = assay_results

    # Save raw results
    output_dir = Path(__file__).parent

    with open(output_dir / "encode_validation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nRaw results saved to: {output_dir / 'encode_validation_results.json'}")

    # Generate ordered code
    ordered_code = generate_ordered_code(all_results)
    with open(output_dir / "ordered_vocabularies.py", "w") as f:
        f.write(ordered_code)
    print(f"Ordered vocabularies saved to: {output_dir / 'ordered_vocabularies.py'}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nWorking API parameter paths:")
    for name, path in all_results["working_paths"].items():
        print(f"  {name}: {path or 'NOT FOUND'}")

    print("\nTop 10 assays by popularity:")
    top_assays = sorted(assay_results, key=lambda x: x["count"], reverse=True)[:10]
    for a in top_assays:
        print(f"  {a['count']:6d}  {a['key']}")

    zero_count = len([a for a in assay_results if a["count"] == 0])
    error_count = len([a for a in assay_results if a["count"] < 0])
    print(f"\nAssays with zero results: {zero_count}")
    print(f"Assays with errors: {error_count}")

    print("\n" + "=" * 60)
    print("DONE - Review ordered_vocabularies.py and update src/ui/vocabularies.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
