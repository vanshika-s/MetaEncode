#!/usr/bin/env python3
"""Compare vocabularies.py with actual ENCODE API values.

This script compares our vocabulary definitions with the actual values
available in the ENCODE API schema.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.vocabularies import (
    ASSAY_TYPES,
    BODY_PARTS,
    COMMON_LABS,
    DEVELOPMENTAL_STAGES,
    HISTONE_MODIFICATIONS,
    ORGANISMS,
)

# Official ENCODE assay_term_name values from mixins.json (dev branch)
# Source: https://github.com/ENCODE-DCC/encoded/blob/dev/src/encoded/schemas/mixins.json
ENCODE_ASSAY_TYPES = {
    "3' RACE",
    "4C",
    "5' RACE",
    "5' RLM RACE",
    "5C",
    "ATAC-seq",
    "Bru-seq",
    "BruChase-seq",
    "BruUV-seq",
    "CAGE",
    "capture Hi-C",
    "ChIA-PET",
    "ChIP-seq",
    "Circulome-seq",
    "Clone-seq",
    "comparative genomic hybridization by array",
    "CRISPR genome editing followed by RNA-seq",
    "CRISPRi followed by RNA-seq",
    "CUT&RUN",
    "CUT&Tag",
    "direct RNA-seq",
    "DNA methylation profiling by array assay",
    "DNA-PET",
    "DNase-seq",
    "eCLIP",
    "FAIRE-seq",
    "genetic modification followed by DNase-seq",
    "genomic perturbation followed by RT-qPCR",
    "genotype phasing by HiC",
    "GRO-cap",
    "GRO-seq",
    "HiC",
    "iCLIP",
    "icLASER",
    "icSHAPE",
    "LC/MS label-free quantitative proteomics",
    "LC-MS/MS isobaric label quantitative proteomics",
    "long read RNA-seq",
    "long read single-cell RNA-seq",
    "MeDIP-seq",
    "microRNA counts",
    "microRNA-seq",
    "Mint-ChIP-seq",
    "MNase-seq",
    "MRE-seq",
    "PAS-seq",
    "PLAC-seq",
    "polyA plus RNA-seq",
    "polyA minus RNA-seq",
    "PRO-cap",
    "PRO-seq",
    "protein sequencing by tandem mass spectrometry assay",
    "RAMPAGE",
    "Repli-chip",
    "Repli-seq",
    "Ribo-seq",
    "RIP-chip",
    "RIP-seq",
    "RNA Bind-n-Seq",
    "RNA-PET",
    "RNA-seq",
    "RRBS",
    "seqFISH",
    "shRNA knockdown followed by RNA-seq",
    "single-cell RNA sequencing assay",
    "single-nucleus ATAC-seq",
    "siRNA knockdown followed by RNA-seq",
    "small RNA-seq",
    "SPRITE",
    "SPRITE-IP",
    "Switchgear",
    "TAB-seq",
    "transcription profiling by array assay",
    "whole genome sequencing assay",
    "whole-genome shotgun bisulfite sequencing",
}

# Official ENCODE organisms
ENCODE_ORGANISMS = {"human", "mouse", "fly", "worm"}

# ENCODE Tier 1/2 cell types (commonly used)
ENCODE_CELL_TYPES = {
    # Tier 1
    "GM12878",
    "K562",
    "H1-hESC",
    "H1",
    # Tier 2
    "A549",
    "HeLa-S3",
    "HepG2",
    "HUVEC",
    "IMR-90",
    "MCF-7",
    "SK-N-SH",
    # Common Tier 3
    "HEK293",
    "PC-3",
    "LNCaP",
    "NHEK",
    "Jurkat",
    "U2OS",
    "Caco-2",
    "293T",
}


def compare_assay_types():
    """Compare assay types in vocabularies.py with ENCODE schema."""
    print("=" * 70)
    print("ASSAY TYPES COMPARISON")
    print("=" * 70)

    our_assays = set(ASSAY_TYPES.keys())

    # In ENCODE but not in our vocab
    in_encode_not_ours = ENCODE_ASSAY_TYPES - our_assays
    # In our vocab but not in ENCODE
    in_ours_not_encode = our_assays - ENCODE_ASSAY_TYPES

    print(f"\nOur vocabulary has: {len(our_assays)} assay types")
    print(f"ENCODE schema has: {len(ENCODE_ASSAY_TYPES)} assay types")

    if in_encode_not_ours:
        print(f"\n### MISSING from our vocabulary ({len(in_encode_not_ours)}):")
        for assay in sorted(in_encode_not_ours):
            print(f"  + {assay}")

    if in_ours_not_encode:
        print(f"\n### INVALID in our vocabulary ({len(in_ours_not_encode)}):")
        print("### (These do NOT exist in ENCODE schema)")
        for assay in sorted(in_ours_not_encode):
            print(f"  - {assay}")

    # Check case sensitivity issues
    our_lower = {a.lower() for a in our_assays}
    encode_lower = {a.lower() for a in ENCODE_ASSAY_TYPES}
    case_issues = []
    for our in our_assays:
        for enc in ENCODE_ASSAY_TYPES:
            if our.lower() == enc.lower() and our != enc:
                case_issues.append((our, enc))

    if case_issues:
        print(f"\n### CASE SENSITIVITY ISSUES:")
        for ours, theirs in case_issues:
            print(f"  Ours: '{ours}' -> ENCODE: '{theirs}'")


def compare_organisms():
    """Compare organisms."""
    print("\n" + "=" * 70)
    print("ORGANISMS COMPARISON")
    print("=" * 70)

    our_orgs = set(ORGANISMS.keys())

    print(f"\nOur vocabulary has: {len(our_orgs)} organisms")
    print(f"ENCODE has: {len(ENCODE_ORGANISMS)} main organisms")

    # Check coverage
    in_encode_not_ours = ENCODE_ORGANISMS - our_orgs
    in_ours_not_encode = our_orgs - ENCODE_ORGANISMS

    if in_encode_not_ours:
        print(f"\n### MISSING from our vocabulary:")
        for org in sorted(in_encode_not_ours):
            print(f"  + {org}")

    if in_ours_not_encode:
        print(f"\n### In our vocab but NOT in ENCODE:")
        for org in sorted(in_ours_not_encode):
            print(f"  - {org}")

    if not in_encode_not_ours and not in_ours_not_encode:
        print("\n  PERFECT MATCH!")


def analyze_cell_types():
    """Analyze cell types in our body parts vocabulary."""
    print("\n" + "=" * 70)
    print("CELL TYPES ANALYSIS")
    print("=" * 70)

    # Get all tissues/cells from our vocabulary
    our_cells = set()
    for bp_info in BODY_PARTS.values():
        for tissue in bp_info["tissues"]:
            our_cells.add(tissue)

    print(f"\nOur vocabulary has: {len(our_cells)} unique tissue/cell types")
    print(f"ENCODE Tier 1/2 cell lines: {len(ENCODE_CELL_TYPES)}")

    # Check how many ENCODE cell lines we have
    encode_cells_found = our_cells & ENCODE_CELL_TYPES
    encode_cells_missing = ENCODE_CELL_TYPES - our_cells

    print(f"\n### ENCODE cell lines we have: {len(encode_cells_found)}")
    for cell in sorted(encode_cells_found):
        print(f"  ✓ {cell}")

    if encode_cells_missing:
        print(f"\n### ENCODE cell lines we're MISSING: {len(encode_cells_missing)}")
        for cell in sorted(encode_cells_missing):
            print(f"  + {cell}")


def analyze_developmental_stages():
    """Analyze developmental stages approach."""
    print("\n" + "=" * 70)
    print("DEVELOPMENTAL STAGES ANALYSIS")
    print("=" * 70)

    print(f"\nOur vocabulary has: {len(DEVELOPMENTAL_STAGES)} developmental stages")

    # ENCODE life_stage values
    encode_life_stages = {
        "mouse": ["embryonic", "postnatal", "adult", "unknown"],
        "fly": [
            "embryonic",
            "larva",
            "first instar larva",
            "second instar larva",
            "third instar larva",
            "wandering third instar larva",
            "prepupa",
            "pupa",
            "adult",
        ],
        "worm": [
            "early embryonic",
            "midembryonic",
            "late embryonic",
            "L1 larva",
            "L2 larva",
            "L3 larva",
            "L4 larva",
            "dauer",
            "adult",
            "mixed stage",
        ],
    }

    print("\n### ENCODE life_stage values (API searchable):")
    for species, stages in encode_life_stages.items():
        print(f"\n  {species.upper()}:")
        for stage in stages:
            print(f"    - {stage}")

    print("\n### IMPORTANT NOTE:")
    print("  Specific ages like 'P60', 'E14.5', '8 weeks' are NOT direct")
    print("  search parameters. They are stored in 'model_organism_age' field")
    print("  and must be searched via:")
    print("    1. Description text search")
    print("    2. Post-filtering on fetched results")
    print("    3. ML-based similarity matching")

    # Count mouse vs non-mouse stages
    mouse_stages = [
        k for k, v in DEVELOPMENTAL_STAGES.items() if v.get("species") == "mouse"
    ]
    human_stages = [
        k for k, v in DEVELOPMENTAL_STAGES.items() if v.get("species") == "human"
    ]

    print(f"\n### Our vocabulary breakdown:")
    print(f"  Mouse stages: {len(mouse_stages)}")
    print(f"  Human stages: {len(human_stages)}")


def main():
    """Run all comparisons."""
    print("\n" + "=" * 70)
    print("VOCABULARY COMPARISON WITH ACTUAL ENCODE DATA")
    print("=" * 70)

    compare_assay_types()
    compare_organisms()
    analyze_cell_types()
    analyze_developmental_stages()

    print("\n" + "=" * 70)
    print("SUMMARY OF ISSUES")
    print("=" * 70)

    print(
        """
### HIGH PRIORITY FIXES:

1. ASSAY TYPES:
   - Fix case sensitivity (Hi-C -> HiC, etc.)
   - Remove assays not in ENCODE schema
   - Add missing common assays from ENCODE

2. SEARCH IMPLEMENTATION:
   - Change from client.search() to client.fetch_experiments()
   - Use proper API parameters (assay_term_name, biosample_ontology.term_name)
   - Remove developmental stages from API search query

3. DEVELOPMENTAL STAGES:
   - Do NOT use as API search parameter
   - Use life_stage for broad categories (embryonic, postnatal, adult)
   - Use description text filtering for specific ages (P60, E14.5)
   - Consider ML similarity for related time points

4. CELL TYPES:
   - Add missing ENCODE Tier 1/2 cell lines
   - Order by ENCODE Tier for common options
"""
    )


if __name__ == "__main__":
    main()
