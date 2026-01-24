# src/ui/vocabularies.py
"""Vocabulary definitions for ENCODE dataset search and filtering.

ARCHITECTURE: Single Source of Truth
====================================
All vocabulary values are dynamically loaded from `scripts/encode_facets_raw.json`,
which is the authoritative source derived from ENCODE API. This ensures:
- No drift between JSON and Python
- No fabricated values (all data exists in ENCODE)
- Automatic popularity ordering (by experiment count)

This module provides:
- Accessor functions for each vocabulary type (load dynamically from JSON)
- Alias mappings for search term normalization (curated for user input)
- Display name mappings for UI (optional shortened names)
- Organism metadata (genome assemblies, etc.)
- Body part groupings (curated for UI organization)

Last updated: 2026-01-23
Source: ENCODE API /search/?type=Experiment
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

# =============================================================================
# JSON DATA LOADING (Single Source of Truth)
# =============================================================================

# Path to the source of truth JSON file
_FACETS_PATH = Path(__file__).parent.parent.parent / "data" / "encode_facets_raw.json"

# Cached facets data (module-level cache)
_facets_data: dict | None = None


def _load_facets() -> dict:
    """Load and cache facets data from JSON.

    Returns:
        Dictionary containing field_counts and metadata from ENCODE.

    Raises:
        FileNotFoundError: If encode_facets_raw.json doesn't exist.
        json.JSONDecodeError: If JSON is malformed.
    """
    global _facets_data
    if _facets_data is None:
        with open(_FACETS_PATH) as f:
            _facets_data = json.load(f)
    return _facets_data


def reload_facets() -> None:
    """Force reload of facets data (useful after regenerating JSON)."""
    global _facets_data
    _facets_data = None
    _load_facets()


# =============================================================================
# SLIM TYPE CONFIGURATION
# =============================================================================

# Configuration for all four ENCODE slim types for uniform handling
SLIM_TYPES: dict[str, dict[str, str]] = {
    "organ": {
        "json_key": "organ_to_biosamples",
        "display_prefix": "Organ System",
        "description": "Anatomical organ classification (brain, heart, liver)",
    },
    "cell": {
        "json_key": "cell_to_biosamples",
        "display_prefix": "Cell Type",
        "description": "Cell type classification (T cell, stem cell, epithelial cell)",
    },
    "developmental": {
        "json_key": "developmental_to_biosamples",
        "display_prefix": "Germ Layer",
        "description": "Embryonic germ layer origin (mesoderm, ectoderm, endoderm)",
    },
    "system": {
        "json_key": "system_to_biosamples",
        "display_prefix": "Body System",
        "description": "Body system classification (nervous system, immune system)",
    },
}


# =============================================================================
# ACCESSOR FUNCTIONS (Load from JSON, ordered by popularity)
# =============================================================================


@lru_cache(maxsize=1)
def get_assay_types() -> list[tuple[str, int]]:
    """Return assay types ordered by experiment count (popularity).

    Returns:
        List of (assay_name, count) tuples, most popular first.

    Example:
        >>> assays = get_assay_types()
        >>> assays[0]  # Most popular
        ('ChIP-seq', 12569)
    """
    data = _load_facets()
    return [
        (item["key"], item["count"]) for item in data["field_counts"]["assay_term_name"]
    ]


@lru_cache(maxsize=1)
def get_biosamples() -> list[tuple[str, int]]:
    """Return biosamples ordered by experiment count (popularity).

    Returns:
        List of (biosample_name, count) tuples, most popular first.
    """
    data = _load_facets()
    return [
        (item["key"], item["count"])
        for item in data["field_counts"]["biosample_ontology.term_name"]
    ]


@lru_cache(maxsize=1)
def get_targets() -> list[tuple[str, int]]:
    """Return ChIP-seq targets ordered by experiment count (popularity).

    Returns:
        List of (target_name, count) tuples, most popular first.
    """
    data = _load_facets()
    return [
        (item["key"], item["count"]) for item in data["field_counts"]["target.label"]
    ]


@lru_cache(maxsize=1)
def get_life_stages() -> list[tuple[str, int]]:
    """Return ACTUAL life stages from ENCODE, ordered by experiment count.

    These are the real ENCODE life_stage values, NOT fabricated developmental
    stages like "E10.5" or "P56" which don't exist in the ENCODE API.

    Returns:
        List of (life_stage, count) tuples, most popular first.

    Example:
        >>> stages = get_life_stages()
        >>> stages[:3]
        [('adult', 25196), ('embryonic', 9573), ('unknown', 4861)]
    """
    data = _load_facets()
    return [(item["key"], item["count"]) for item in data["field_counts"]["life_stage"]]


@lru_cache(maxsize=1)
def get_labs() -> list[tuple[str, int]]:
    """Return labs ordered by experiment count (popularity).

    Returns:
        List of (lab_name, count) tuples, most popular first.
    """
    data = _load_facets()
    return [(item["key"], item["count"]) for item in data["field_counts"]["lab.title"]]


@lru_cache(maxsize=1)
def get_organisms_from_json() -> list[tuple[str, int]]:
    """Return organisms (scientific names) ordered by experiment count.

    Returns:
        List of (scientific_name, count) tuples, most popular first.

    Note:
        Use ORGANISMS dict for common name -> scientific name mapping.
    """
    data = _load_facets()
    return [(item["key"], item["count"]) for item in data["field_counts"]["organism"]]


def get_total_experiments() -> int:
    """Return total number of experiments in the ENCODE database.

    Returns:
        Total experiment count from JSON metadata.
    """
    data = _load_facets()
    return int(data.get("total_experiments", 0))


def get_facets_timestamp() -> str:
    """Return timestamp when facets were last fetched.

    Returns:
        ISO format timestamp string.
    """
    data = _load_facets()
    return str(data.get("fetched_at", "unknown"))


# =============================================================================
# ORGAN SYSTEMS (from ENCODE's organ_slims ontology)
# =============================================================================


@lru_cache(maxsize=1)
def get_organ_systems() -> list[tuple[str, int]]:
    """Return organ systems from ENCODE's organ_slims, ordered by total experiments.

    Uses UBERON ontology-derived organ classifications from biosample_ontology.organ_slims.
    Each organ system maps to multiple biosamples.

    Returns:
        List of (organ_name, total_experiment_count) tuples, ordered by experiment count.

    Example:
        >>> organs = get_organ_systems()
        >>> organs[0]
        ('bodily fluid', 6193)
    """
    data = _load_facets()
    organ_data = data.get("organ_to_biosamples", {})
    # Calculate total experiments per organ
    organ_totals = []
    for organ, biosamples in organ_data.items():
        total = sum(item["count"] for item in biosamples)
        organ_totals.append((organ, total))
    return sorted(organ_totals, key=lambda x: -x[1])


def get_organ_system_names() -> list[str]:
    """Return organ system names ordered by popularity.

    Returns:
        List of organ names, most experiments first.
    """
    return [name for name, count in get_organ_systems()]


@lru_cache(maxsize=64)
def get_biosamples_for_organ(organ: str) -> list[tuple[str, int]]:
    """Return biosamples for an organ system, ordered by experiment count.

    Args:
        organ: Organ system name (e.g., "brain", "heart", "liver").

    Returns:
        List of (biosample_name, experiment_count) tuples, ordered by count.
        Empty list if organ not found.

    Example:
        >>> brain = get_biosamples_for_organ("brain")
        >>> brain[0]
        ('dorsolateral prefrontal cortex', 605)
    """
    data = _load_facets()
    items = data.get("organ_to_biosamples", {}).get(organ, [])
    return [(item["key"], item["count"]) for item in items]


def get_biosample_names_for_organ(organ: str, limit: int | None = None) -> list[str]:
    """Return biosample names for an organ system.

    Args:
        organ: Organ system name (e.g., "brain", "heart").
        limit: Optional limit on number of biosamples to return.

    Returns:
        List of biosample names, most popular first. Empty list if organ not found.
    """
    biosamples = get_biosamples_for_organ(organ)
    if limit:
        biosamples = biosamples[:limit]
    return [name for name, _ in biosamples]


# Display name mappings for organ_slims (optional UI polish)
ORGAN_DISPLAY_NAMES: dict[str, str] = {
    "bodily fluid": "Blood / Bodily Fluid",
    "musculature of body": "Muscle",
    "large intestine": "Large Intestine",
    "arterial blood vessel": "Arteries",
    "connective tissue": "Connective Tissue",
    "skin of body": "Skin",
    "bone element": "Bone",
    "bone marrow": "Bone Marrow",
    "lymph node": "Lymph Nodes",
    "urinary bladder": "Bladder",
    "small intestine": "Small Intestine",
    "exocrine gland": "Exocrine Glands",
    "endocrine gland": "Endocrine Glands",
}


def get_organ_display_name(organ: str) -> str:
    """Get UI-friendly display name for an organ system.

    Args:
        organ: Organ system name from organ_slims (e.g., "bodily fluid").

    Returns:
        Display name (e.g., "Blood / Bodily Fluid"), or title-cased original.
    """
    return ORGAN_DISPLAY_NAMES.get(organ, organ.replace("_", " ").title())


# =============================================================================
# BIOSAMPLE-TO-ORGAN REVERSE MAPPING
# =============================================================================


@lru_cache(maxsize=1)
def build_biosample_to_organs() -> dict[str, list[str]]:
    """Build reverse mapping from biosample term_name to organ systems.

    Since biosamples can belong to multiple organs (e.g., 'omental fat pad'
    belongs to 'adipose tissue', 'connective tissue', and 'gonad'), this
    returns a list of organs for each biosample, ordered by experiment count.

    Returns:
        Dictionary mapping biosample names to list of organ names,
        with organs ordered by their total experiment count (most popular first).

    Example:
        >>> mapping = build_biosample_to_organs()
        >>> mapping.get("cerebellum")
        ['brain']
        >>> mapping.get("omental fat pad")
        ['adipose tissue', 'connective tissue', 'gonad']
    """
    data = _load_facets()
    organ_data = data.get("organ_to_biosamples", {})

    biosample_to_organs: dict[str, list[tuple[str, int]]] = {}

    for organ, biosamples in organ_data.items():
        # Get organ's total experiment count for ordering
        organ_total = sum(item["count"] for item in biosamples)
        for item in biosamples:
            key = item["key"]
            if key not in biosample_to_organs:
                biosample_to_organs[key] = []
            biosample_to_organs[key].append((organ, organ_total))

    # Sort organs by experiment count (most popular first)
    result: dict[str, list[str]] = {}
    for biosample, organ_list in biosample_to_organs.items():
        organ_list.sort(key=lambda x: -x[1])
        result[biosample] = [org for org, _ in organ_list]

    return result


def get_primary_organ_for_biosample(biosample: str) -> str | None:
    """Get the primary (most relevant) organ for a biosample.

    Returns the organ with the most experiments for this biosample.
    This is useful for assigning a single organ category for visualization.

    Args:
        biosample: Biosample term_name (e.g., "cerebellum", "K562").

    Returns:
        Primary organ name (e.g., "brain") or None if not found.

    Example:
        >>> get_primary_organ_for_biosample("cerebellum")
        'brain'
        >>> get_primary_organ_for_biosample("K562")
        'blood'
    """
    mapping = build_biosample_to_organs()
    organs = mapping.get(biosample, [])
    return organs[0] if organs else None


def get_all_organs_for_biosample(biosample: str) -> list[str]:
    """Get all organs associated with a biosample.

    Args:
        biosample: Biosample term_name.

    Returns:
        List of organ names, ordered by experiment count.
        Empty list if biosample not found.

    Example:
        >>> get_all_organs_for_biosample("omental fat pad")
        ['adipose tissue', 'connective tissue', 'gonad']
    """
    mapping = build_biosample_to_organs()
    return mapping.get(biosample, [])


# =============================================================================
# GENERIC SLIM TYPE FUNCTIONS
# =============================================================================


def _get_slim_data(slim_type: str) -> dict:
    """Get raw slim data from JSON by type.

    Args:
        slim_type: One of "organ", "cell", "developmental", "system".

    Returns:
        Dictionary mapping slim values to lists of biosample dicts.

    Raises:
        ValueError: If slim_type is not recognized.
    """
    if slim_type not in SLIM_TYPES:
        raise ValueError(
            f"Unknown slim type: {slim_type}. "
            f"Must be one of: {list(SLIM_TYPES.keys())}"
        )
    data = _load_facets()
    json_key = SLIM_TYPES[slim_type]["json_key"]
    return data.get(json_key, {})


@lru_cache(maxsize=4)
def get_slim_categories(slim_type: str) -> list[tuple[str, int]]:
    """Return slim categories ordered by total experiments.

    Generic accessor for all four slim types.

    Args:
        slim_type: One of "organ", "cell", "developmental", "system".

    Returns:
        List of (category_name, total_experiment_count) tuples,
        ordered by experiment count (most popular first).

    Example:
        >>> cells = get_slim_categories("cell")
        >>> cells[0]
        ('cancer cell', 7042)
        >>> layers = get_slim_categories("developmental")
        >>> layers
        [('mesoderm', 11110), ('endoderm', 7533), ('ectoderm', 5487)]
    """
    slim_data = _get_slim_data(slim_type)
    category_totals = []
    for category, biosamples in slim_data.items():
        total = sum(item["count"] for item in biosamples)
        category_totals.append((category, total))
    return sorted(category_totals, key=lambda x: -x[1])


def get_slim_category_names(slim_type: str) -> list[str]:
    """Return slim category names ordered by popularity.

    Args:
        slim_type: One of "organ", "cell", "developmental", "system".

    Returns:
        List of category names, most experiments first.
    """
    return [name for name, _ in get_slim_categories(slim_type)]


@lru_cache(maxsize=128)
def get_biosamples_for_slim(slim_type: str, category: str) -> list[tuple[str, int]]:
    """Return biosamples for a slim category, ordered by experiment count.

    Generic accessor for all four slim types.

    Args:
        slim_type: One of "organ", "cell", "developmental", "system".
        category: The slim category (e.g., "brain", "T cell", "ectoderm").

    Returns:
        List of (biosample_name, experiment_count) tuples, ordered by count.
        Empty list if category not found.

    Example:
        >>> tcells = get_biosamples_for_slim("cell", "T cell")
        >>> tcells[0]
        ('CD4-positive, alpha-beta T cell', ...)
    """
    slim_data = _get_slim_data(slim_type)
    items = slim_data.get(category, [])
    return [(item["key"], item["count"]) for item in items]


@lru_cache(maxsize=4)
def build_biosample_to_slim(slim_type: str) -> dict[str, list[str]]:
    """Build reverse mapping from biosample to slim categories.

    Generic function for all four slim types.

    Args:
        slim_type: One of "organ", "cell", "developmental", "system".

    Returns:
        Dict mapping biosample names to list of slim categories,
        ordered by category experiment count (most popular first).

    Example:
        >>> mapping = build_biosample_to_slim("developmental")
        >>> mapping.get("cerebellum")
        ['ectoderm']
    """
    slim_data = _get_slim_data(slim_type)

    biosample_to_categories: dict[str, list[tuple[str, int]]] = {}

    for category, biosamples in slim_data.items():
        category_total = sum(item["count"] for item in biosamples)
        for item in biosamples:
            key = item["key"]
            if key not in biosample_to_categories:
                biosample_to_categories[key] = []
            biosample_to_categories[key].append((category, category_total))

    result: dict[str, list[str]] = {}
    for biosample, cat_list in biosample_to_categories.items():
        cat_list.sort(key=lambda x: -x[1])
        result[biosample] = [cat for cat, _ in cat_list]

    return result


def get_primary_slim_for_biosample(slim_type: str, biosample: str) -> str | None:
    """Get the primary (most relevant) slim category for a biosample.

    Generic function for all four slim types.

    Args:
        slim_type: One of "organ", "cell", "developmental", "system".
        biosample: Biosample term_name.

    Returns:
        Primary category name or None if not found.

    Example:
        >>> get_primary_slim_for_biosample("developmental", "cerebellum")
        'ectoderm'
        >>> get_primary_slim_for_biosample("system", "heart")
        'circulatory system'
    """
    mapping = build_biosample_to_slim(slim_type)
    categories = mapping.get(biosample, [])
    return categories[0] if categories else None


def get_all_slims_for_biosample(slim_type: str, biosample: str) -> list[str]:
    """Get all slim categories for a biosample.

    Args:
        slim_type: One of "organ", "cell", "developmental", "system".
        biosample: Biosample term_name.

    Returns:
        List of category names, ordered by experiment count.
    """
    mapping = build_biosample_to_slim(slim_type)
    return mapping.get(biosample, [])


# =============================================================================
# CELL SLIMS (Cell type classifications)
# =============================================================================


@lru_cache(maxsize=1)
def get_cell_types() -> list[tuple[str, int]]:
    """Return cell types from ENCODE's cell_slims, ordered by total experiments.

    Returns:
        List of (cell_type_name, total_experiment_count) tuples.

    Example:
        >>> cells = get_cell_types()
        >>> cells[0]
        ('cancer cell', 7042)
    """
    return get_slim_categories("cell")


def get_cell_type_names() -> list[str]:
    """Return cell type names ordered by popularity."""
    return [name for name, _ in get_cell_types()]


@lru_cache(maxsize=64)
def get_biosamples_for_cell_type(cell_type: str) -> list[tuple[str, int]]:
    """Return biosamples for a cell type, ordered by experiment count.

    Args:
        cell_type: Cell type name (e.g., "T cell", "stem cell").

    Returns:
        List of (biosample_name, experiment_count) tuples.
    """
    return get_biosamples_for_slim("cell", cell_type)


def get_primary_cell_type_for_biosample(biosample: str) -> str | None:
    """Get the primary cell type for a biosample.

    Args:
        biosample: Biosample term_name.

    Returns:
        Primary cell type or None if not found.
    """
    return get_primary_slim_for_biosample("cell", biosample)


@lru_cache(maxsize=1)
def build_biosample_to_cell_types() -> dict[str, list[str]]:
    """Build reverse mapping from biosample to cell types."""
    return build_biosample_to_slim("cell")


# Display name mappings for cell_slims
CELL_TYPE_DISPLAY_NAMES: dict[str, str] = {
    "hematopoietic cell": "Blood/Immune Cells",
    "epithelial cell": "Epithelial Cells",
    "mesenchymal cell": "Mesenchymal Cells",
    "neural cell": "Neural Cells",
    "connective tissue cell": "Connective Tissue Cells",
    "cancer cell": "Cancer Cells",
    "stem cell": "Stem Cells",
    "induced pluripotent stem cell": "iPSCs",
    "embryonic cell": "Embryonic Cells",
}


def get_cell_type_display_name(cell_type: str) -> str:
    """Get UI-friendly display name for a cell type."""
    return CELL_TYPE_DISPLAY_NAMES.get(cell_type, cell_type.replace("_", " ").title())


# =============================================================================
# DEVELOPMENTAL SLIMS (Embryonic germ layers)
# =============================================================================


@lru_cache(maxsize=1)
def get_developmental_layers() -> list[tuple[str, int]]:
    """Return developmental/germ layers from ENCODE's developmental_slims.

    These represent the three primary germ layers from embryonic development.

    Returns:
        List of (layer_name, total_experiment_count) tuples.

    Example:
        >>> layers = get_developmental_layers()
        >>> layers
        [('mesoderm', 11110), ('endoderm', 7533), ('ectoderm', 5487)]
    """
    return get_slim_categories("developmental")


def get_developmental_layer_names() -> list[str]:
    """Return developmental layer names ordered by popularity."""
    return [name for name, _ in get_developmental_layers()]


@lru_cache(maxsize=64)
def get_biosamples_for_developmental_layer(layer: str) -> list[tuple[str, int]]:
    """Return biosamples for a developmental layer (e.g., mesoderm).

    Args:
        layer: Germ layer name (mesoderm, ectoderm, or endoderm).

    Returns:
        List of (biosample_name, experiment_count) tuples.
    """
    return get_biosamples_for_slim("developmental", layer)


def get_primary_developmental_layer_for_biosample(biosample: str) -> str | None:
    """Get the primary developmental layer for a biosample.

    Args:
        biosample: Biosample term_name.

    Returns:
        Primary germ layer or None if not found.
    """
    return get_primary_slim_for_biosample("developmental", biosample)


@lru_cache(maxsize=1)
def build_biosample_to_developmental_layers() -> dict[str, list[str]]:
    """Build reverse mapping from biosample to developmental layers."""
    return build_biosample_to_slim("developmental")


# Display name mappings for developmental_slims
DEVELOPMENTAL_DISPLAY_NAMES: dict[str, str] = {
    "mesoderm": "Mesoderm (Middle Layer)",
    "ectoderm": "Ectoderm (Outer Layer)",
    "endoderm": "Endoderm (Inner Layer)",
}


def get_developmental_display_name(layer: str) -> str:
    """Get UI-friendly display name for a developmental layer."""
    return DEVELOPMENTAL_DISPLAY_NAMES.get(layer, layer.title())


# =============================================================================
# SYSTEM SLIMS (Body systems)
# =============================================================================


@lru_cache(maxsize=1)
def get_body_systems() -> list[tuple[str, int]]:
    """Return body systems from ENCODE's system_slims, ordered by experiments.

    Returns:
        List of (system_name, total_experiment_count) tuples.

    Example:
        >>> systems = get_body_systems()
        >>> systems[0]
        ('immune system', 7233)
    """
    return get_slim_categories("system")


def get_body_system_names() -> list[str]:
    """Return body system names ordered by popularity."""
    return [name for name, _ in get_body_systems()]


@lru_cache(maxsize=64)
def get_biosamples_for_body_system(system: str) -> list[tuple[str, int]]:
    """Return biosamples for a body system (e.g., nervous system).

    Args:
        system: Body system name (e.g., "immune system", "nervous system").

    Returns:
        List of (biosample_name, experiment_count) tuples.
    """
    return get_biosamples_for_slim("system", system)


def get_primary_body_system_for_biosample(biosample: str) -> str | None:
    """Get the primary body system for a biosample.

    Args:
        biosample: Biosample term_name.

    Returns:
        Primary body system or None if not found.
    """
    return get_primary_slim_for_biosample("system", biosample)


@lru_cache(maxsize=1)
def build_biosample_to_body_systems() -> dict[str, list[str]]:
    """Build reverse mapping from biosample to body systems."""
    return build_biosample_to_slim("system")


# Display name mappings for system_slims
SYSTEM_DISPLAY_NAMES: dict[str, str] = {
    "central nervous system": "Central Nervous System",
    "peripheral nervous system": "Peripheral Nervous System",
    "immune system": "Immune System",
    "musculature": "Muscular System",
    "circulatory system": "Circulatory System",
    "respiratory system": "Respiratory System",
    "digestive system": "Digestive System",
    "excretory system": "Excretory System",
    "reproductive system": "Reproductive System",
    "integumental system": "Integumentary System",
    "skeletal system": "Skeletal System",
    "endocrine system": "Endocrine System",
    "exocrine system": "Exocrine System",
    "sensory system": "Sensory System",
}


def get_system_display_name(system: str) -> str:
    """Get UI-friendly display name for a body system."""
    return SYSTEM_DISPLAY_NAMES.get(system, system.replace("_", " ").title())


# =============================================================================
# SLIM TYPE DISPLAY NAME DISPATCHER
# =============================================================================


def get_slim_display_name(slim_type: str, value: str) -> str:
    """Get UI-friendly display name for any slim type value.

    Args:
        slim_type: One of "organ", "cell", "developmental", "system".
        value: The slim value to get display name for.

    Returns:
        Display name for the value.
    """
    dispatch = {
        "organ": get_organ_display_name,
        "cell": get_cell_type_display_name,
        "developmental": get_developmental_display_name,
        "system": get_system_display_name,
    }
    if slim_type in dispatch:
        return dispatch[slim_type](value)
    return value.replace("_", " ").title()


# =============================================================================
# CONVENIENCE FUNCTIONS FOR UI
# =============================================================================


def get_assay_type_names() -> list[str]:
    """Return list of assay type names ordered by popularity.

    Use this for populating dropdowns without count information.
    """
    return [name for name, count in get_assay_types()]


def get_biosample_names(limit: int | None = None) -> list[str]:
    """Return list of biosample names ordered by popularity.

    Args:
        limit: Optional limit on number of biosamples to return.

    Returns:
        List of biosample names, most popular first.
    """
    biosamples = get_biosamples()
    if limit:
        biosamples = biosamples[:limit]
    return [name for name, count in biosamples]


def get_target_names(limit: int | None = None) -> list[str]:
    """Return list of target names ordered by popularity.

    Args:
        limit: Optional limit on number of targets to return.

    Returns:
        List of target names (histone marks, TFs), most popular first.
    """
    targets = get_targets()
    if limit:
        targets = targets[:limit]
    return [name for name, count in targets]


def get_life_stage_names() -> list[str]:
    """Return list of life stage names ordered by popularity."""
    return [name for name, count in get_life_stages()]


def get_lab_names(limit: int | None = None) -> list[str]:
    """Return list of lab names ordered by popularity.

    Args:
        limit: Optional limit on number of labs to return.

    Returns:
        List of lab names, most popular first.
    """
    labs = get_labs()
    if limit:
        labs = labs[:limit]
    return [name for name, count in labs]


# =============================================================================
# DISPLAY NAME MAPPINGS (Optional UI-friendly shortened names)
# Keys MUST exist in ENCODE data (validated against JSON)
# =============================================================================

ASSAY_DISPLAY_NAMES: dict[str, str] = {
    # Shortened names for long assay types
    "single-cell RNA sequencing assay": "scRNA-seq",
    "single-nucleus ATAC-seq": "snATAC-seq",
    "whole-genome shotgun bisulfite sequencing": "WGBS",
    "shRNA knockdown followed by RNA-seq": "shRNA knockdown RNA-seq",
    "CRISPR genome editing followed by RNA-seq": "CRISPR RNA-seq",
    "CRISPRi followed by RNA-seq": "CRISPRi RNA-seq",
    "siRNA knockdown followed by RNA-seq": "siRNA knockdown RNA-seq",
    "transcription profiling by array assay": "Expression Array",
    "DNA methylation profiling by array assay": "Methylation Array",
    "comparative genomic hybridization by array": "CGH Array",
    "whole genome sequencing assay": "WGS",
    "genetic modification followed by DNase-seq": "CRISPR-DNase",
    "protein sequencing by tandem mass spectrometry assay": "MS/MS Proteomics",
    "long read single-cell RNA-seq": "long read scRNA-seq",
}


def get_assay_display_name(assay: str) -> str:
    """Get display name for an assay type.

    Args:
        assay: Assay type name from ENCODE.

    Returns:
        Shortened display name if available, otherwise original name.
    """
    return ASSAY_DISPLAY_NAMES.get(assay, assay)


def format_assay_with_count(assay: str, count: int) -> str:
    """Format assay type with experiment count for display.

    Args:
        assay: Assay type name from ENCODE.
        count: Number of experiments.

    Returns:
        Formatted string like "ChIP-seq (12,569 experiments)".
    """
    display_name = get_assay_display_name(assay)
    return f"{display_name} ({count:,} experiments)"


# =============================================================================
# ALIAS MAPPINGS (For search term normalization)
# These help match user input to ENCODE vocabulary
# =============================================================================

ASSAY_ALIASES: dict[str, list[str]] = {
    "ChIP-seq": ["chip", "chipseq", "chip-seq", "chromatin immunoprecipitation"],
    "RNA-seq": ["rna", "rnaseq", "rna-seq", "transcriptome", "expression"],
    "ATAC-seq": ["atac", "atacseq", "atac-seq", "chromatin accessibility"],
    "DNase-seq": ["dnase", "dnaseseq", "dnase-seq", "dhs"],
    "HiC": ["hic", "hi-c", "Hi-C", "chromatin conformation", "3d genome"],
    "whole-genome shotgun bisulfite sequencing": [
        "wgbs",
        "bisulfite",
        "methylation",
        "dna methylation",
    ],
    "eCLIP": ["eclip", "clip", "rna binding"],
    "single-cell RNA sequencing assay": ["scrna", "scrnaseq", "single cell rna"],
    "single-nucleus ATAC-seq": ["snatac", "snatacsep", "single nucleus atac"],
    "polyA plus RNA-seq": ["polya", "mrna-seq", "mrna"],
    "microRNA-seq": ["mirna", "microrna", "mir-seq"],
}

HISTONE_ALIASES: dict[str, list[str]] = {
    "H3K27ac": ["h3k27ac", "k27ac", "h3 k27ac", "acetylation k27"],
    "H3K4me1": ["h3k4me1", "k4me1", "h3 k4me1", "monomethyl k4"],
    "H3K4me3": ["h3k4me3", "k4me3", "h3 k4me3", "trimethyl k4"],
    "H3K27me3": ["h3k27me3", "k27me3", "h3 k27me3", "polycomb"],
    "H3K9me3": ["h3k9me3", "k9me3", "h3 k9me3", "heterochromatin"],
    "H3K36me3": ["h3k36me3", "k36me3", "h3 k36me3", "gene body"],
    "CTCF": ["ctcf", "insulator", "boundary"],
}


def normalize_search_term(term: str, alias_dict: dict[str, list[str]]) -> str | None:
    """Normalize a search term using alias mappings.

    Args:
        term: User input search term.
        alias_dict: Dictionary mapping canonical terms to aliases.

    Returns:
        Canonical term if found in aliases, None otherwise.
    """
    term_lower = term.lower().strip()
    for canonical, aliases in alias_dict.items():
        if term_lower == canonical.lower() or term_lower in [
            a.lower() for a in aliases
        ]:
            return canonical
    return None


# =============================================================================
# ORGANISM METADATA (Dynamic from JSON + assembly info for known organisms)
# =============================================================================

# Assembly info for well-known model organisms (not available in ENCODE API)
# This is metadata about genome builds, not a filter restriction
ORGANISM_ASSEMBLIES: dict[str, dict[str, str]] = {
    "Homo sapiens": {
        "common_name": "human",
        "short_name": "Human",
        "assembly": "hg38",
        "alt_assembly": "GRCh38",
    },
    "Mus musculus": {
        "common_name": "mouse",
        "short_name": "Mouse",
        "assembly": "mm10",
        "alt_assembly": "GRCm38",
    },
    "Drosophila melanogaster": {
        "common_name": "fly",
        "short_name": "D. melanogaster",
        "assembly": "dm6",
        "alt_assembly": "BDGP6",
    },
    "Caenorhabditis elegans": {
        "common_name": "worm",
        "short_name": "C. elegans",
        "assembly": "ce11",
        "alt_assembly": "WBcel235",
    },
}


def get_organisms() -> list[tuple[str, int]]:
    """Return organisms from ENCODE, ordered by experiment count.

    Returns:
        List of (scientific_name, count) tuples, most popular first.
    """
    data = _load_facets()
    return [(item["key"], item["count"]) for item in data["field_counts"]["organism"]]


def get_organism_names(limit: int | None = None) -> list[str]:
    """Return list of organism scientific names ordered by popularity.

    Args:
        limit: Optional limit on number of organisms to return.

    Returns:
        List of scientific names, most popular first.
    """
    organisms = get_organisms()
    if limit:
        organisms = organisms[:limit]
    return [name for name, _ in organisms]


def get_organism_common_name(scientific_name: str) -> str | None:
    """Get common name for a scientific name (if known).

    Args:
        scientific_name: Scientific name (e.g., "Homo sapiens").

    Returns:
        Common name (e.g., "human") or None if not a known organism.
    """
    info = ORGANISM_ASSEMBLIES.get(scientific_name)
    return info["common_name"] if info else None


def get_organism_scientific_name(common_or_scientific: str) -> str:
    """Get scientific name for filtering ENCODE API.

    Args:
        common_or_scientific: Common name (e.g., "human") or scientific name.

    Returns:
        Scientific name (e.g., "Homo sapiens").
    """
    # Check if it's already a scientific name
    if common_or_scientific in ORGANISM_ASSEMBLIES:
        return common_or_scientific

    # Check if it's a common name
    for sci_name, info in ORGANISM_ASSEMBLIES.items():
        if info["common_name"] == common_or_scientific:
            return sci_name

    # Return as-is (might be a scientific name not in our assembly dict)
    return common_or_scientific


def get_organism_display(organism: str) -> str:
    """Get display name with genome assembly for an organism.

    Args:
        organism: Common name (e.g., "human") or scientific name.

    Returns:
        Display string like "Human [hg38]" or just the scientific name.
    """
    # First, resolve to scientific name
    sci_name = get_organism_scientific_name(organism)

    # Check if we have assembly info
    if sci_name in ORGANISM_ASSEMBLIES:
        info = ORGANISM_ASSEMBLIES[sci_name]
        return f"{info['short_name']} [{info['assembly']}]"

    # For unknown organisms, just return the scientific name
    return sci_name


def get_all_organisms() -> list[str]:
    """Return list of all organism scientific names from ENCODE.

    Returns:
        List of scientific names ordered by experiment count.
    """
    return get_organism_names()


# Legacy ORGANISMS dict for backward compatibility
# Maps common name -> organism info (same structure as before)
ORGANISMS: dict[str, dict[str, str]] = {
    info["common_name"]: {
        "display_name": f"{info['short_name']} ({sci_name})",
        "scientific_name": sci_name,
        "assembly": info["assembly"],
        "alt_assembly": info.get("alt_assembly", ""),
    }
    for sci_name, info in ORGANISM_ASSEMBLIES.items()
}


# =============================================================================
# TARGET/HISTONE MARK METADATA (Descriptions and categories)
# For display purposes in UI - keys should exist in ENCODE targets
# =============================================================================

HISTONE_MODIFICATIONS: dict[str, dict[str, str]] = {
    # Active histone marks
    "H3K4me3": {
        "full_name": "H3K4me3",
        "description": "Active promoters",
        "category": "promoter",
    },
    "H3K27ac": {
        "full_name": "H3K27ac",
        "description": "Active enhancers and promoters",
        "category": "active",
    },
    "H3K4me1": {
        "full_name": "H3K4me1",
        "description": "Enhancers",
        "category": "enhancer",
    },
    "H3K36me3": {
        "full_name": "H3K36me3",
        "description": "Transcribed gene bodies",
        "category": "transcription",
    },
    "H3K9ac": {
        "full_name": "H3K9ac",
        "description": "Active chromatin",
        "category": "active",
    },
    "H3K4me2": {
        "full_name": "H3K4me2",
        "description": "Active promoters and enhancers",
        "category": "active",
    },
    "H3K79me2": {
        "full_name": "H3K79me2",
        "description": "Active transcription",
        "category": "transcription",
    },
    "H4K20me1": {
        "full_name": "H4K20me1",
        "description": "Active transcription",
        "category": "transcription",
    },
    "H2AFZ": {
        "full_name": "H2A.Z",
        "description": "Promoters and enhancers",
        "category": "active",
    },
    # Repressive histone marks
    "H3K27me3": {
        "full_name": "H3K27me3",
        "description": "Polycomb repression",
        "category": "repressive",
    },
    "H3K9me3": {
        "full_name": "H3K9me3",
        "description": "Heterochromatin",
        "category": "repressive",
    },
    # Transcription factors and other targets
    "CTCF": {
        "full_name": "CTCF",
        "description": "Insulator / chromatin organizer",
        "category": "tf",
    },
    "POLR2A": {
        "full_name": "RNA Polymerase II",
        "description": "Active transcription",
        "category": "tf",
    },
    "EP300": {
        "full_name": "p300",
        "description": "Active enhancers",
        "category": "tf",
    },
    "RAD21": {
        "full_name": "RAD21 (Cohesin)",
        "description": "Chromatin loops",
        "category": "tf",
    },
}


def get_target_description(target: str) -> str | None:
    """Get description for a target/histone mark.

    Args:
        target: Target name (e.g., "H3K27ac", "CTCF").

    Returns:
        Description string or None if not in curated list.
    """
    if target in HISTONE_MODIFICATIONS:
        return HISTONE_MODIFICATIONS[target]["description"]
    return None


def get_all_histone_mods() -> list[str]:
    """Return list of all curated histone modification/target keys."""
    return list(HISTONE_MODIFICATIONS.keys())


# =============================================================================
# BODY PARTS AND ORGAN SYSTEMS (Curated for UI organization)
# Maps organ systems to specific tissues for hierarchical selection
# =============================================================================

BODY_PARTS: dict[str, dict[str, Any]] = {
    "brain": {
        "display_name": "Brain / Nervous System",
        "tissues": [
            "brain",
            "dorsolateral prefrontal cortex",
            "layer of hippocampus",
            "left cerebral cortex",
            "forebrain",
            "midbrain",
            "hindbrain",
            "cerebellum",
            "substantia nigra",
            "head of caudate nucleus",
            "caudate nucleus",
            "putamen",
            "angular gyrus",
            "anterior cingulate cortex",
            "middle frontal area",
            "superior temporal gyrus",
            "spinal cord",
            "tibial nerve",
        ],
        "aliases": [
            "nervous system",
            "cns",
            "central nervous system",
            "neural",
            "cortex",
        ],
    },
    "heart": {
        "display_name": "Heart / Cardiovascular",
        "tissues": [
            "heart",
            "heart left ventricle",
            "heart right ventricle",
            "left cardiac atrium",
            "right cardiac atrium",
            "aorta",
            "coronary artery",
            "thoracic aorta",
        ],
        "aliases": ["cardiovascular", "cardiac", "circulatory"],
    },
    "liver": {
        "display_name": "Liver",
        "tissues": [
            "liver",
            "right lobe of liver",
            "HepG2",
        ],
        "aliases": ["hepatic"],
    },
    "kidney": {
        "display_name": "Kidney",
        "tissues": [
            "kidney",
            "renal cortex",
            "renal medulla",
        ],
        "aliases": ["renal"],
    },
    "lung": {
        "display_name": "Lung / Respiratory",
        "tissues": [
            "lung",
            "upper lobe of left lung",
            "A549",
        ],
        "aliases": ["respiratory", "pulmonary"],
    },
    "blood": {
        "display_name": "Blood / Immune",
        "tissues": [
            "K562",
            "GM12878",
            "CD4-positive, alpha-beta T cell",
            "naive thymus-derived CD4-positive, alpha-beta T cell",
            "T-cell",
            "CD14-positive monocyte",
            "macrophage",
            "B cell",
            "spleen",
            "thymus",
            "bone marrow",
        ],
        "aliases": ["hematopoietic", "immune", "lymphoid", "myeloid"],
    },
    "gut": {
        "display_name": "Gastrointestinal",
        "tissues": [
            "stomach",
            "transverse colon",
            "sigmoid colon",
            "small intestine",
            "esophagus",
            "esophagus muscularis mucosa",
            "HCT116",
        ],
        "aliases": ["digestive", "gastrointestinal", "gi tract", "intestinal"],
    },
    "reproductive": {
        "display_name": "Reproductive",
        "tissues": [
            "ovary",
            "testis",
            "placenta",
            "uterus",
            "prostate",
            "MCF-7",
        ],
        "aliases": ["gonad", "germline"],
    },
    "muscle": {
        "display_name": "Muscle",
        "tissues": [
            "gastrocnemius",
            "gastrocnemius medialis",
            "skeletal muscle tissue",
            "psoas muscle",
        ],
        "aliases": ["muscular", "myogenic"],
    },
    "skin": {
        "display_name": "Skin / Integumentary",
        "tissues": [
            "foreskin keratinocyte",
            "skin of body",
            "keratinocyte",
        ],
        "aliases": ["integumentary", "dermal", "epidermal"],
    },
    "embryonic": {
        "display_name": "Embryonic / Stem Cells",
        "tissues": [
            "H1",
            "H9",
            "WTC11",
            "embryonic facial prominence",
            "limb",
            "whole organism",
        ],
        "aliases": ["stem cell", "pluripotent", "developmental", "esc", "ipsc"],
    },
    "cell_line": {
        "display_name": "Cell Lines",
        "tissues": [
            "K562",
            "HepG2",
            "A549",
            "GM12878",
            "HCT116",
            "MCF-7",
            "SK-N-SH",
            "HEK293",
            "H1",
            "HeLa-S3",
            "WTC11",
            "BLaER1",
            "IMR-90",
            "MEL",
            "H9",
        ],
        "aliases": ["immortalized", "cancer cell line", "transformed"],
    },
}


def get_all_body_parts() -> list[str]:
    """Return list of body part keys."""
    return list(BODY_PARTS.keys())


def get_tissues_for_body_part(body_part: str) -> list[str]:
    """Return list of tissues for a given body part.

    Args:
        body_part: Body part key (e.g., "brain", "heart").

    Returns:
        List of tissue names, or empty list if body part not found.
    """
    if body_part in BODY_PARTS:
        tissues: list[str] = BODY_PARTS[body_part]["tissues"]
        return tissues
    return []


def get_body_part_display_name(body_part: str) -> str:
    """Get display name for a body part.

    Args:
        body_part: Body part key (e.g., "brain").

    Returns:
        Display name (e.g., "Brain / Nervous System").
    """
    if body_part in BODY_PARTS:
        display: str = BODY_PARTS[body_part]["display_name"]
        return display
    return body_part


# =============================================================================
# TISSUE SYNONYMS FOR NLP MATCHING
# =============================================================================

TISSUE_SYNONYMS: dict[str, set[str]] = {
    "cerebellum": {"hindbrain", "metencephalon", "cerebellar"},
    "hindbrain": {"cerebellum", "rhombencephalon", "metencephalon"},
    "hippocampus": {"hippocampal formation", "layer of hippocampus"},
    "cortex": {
        "cerebral cortex",
        "neocortex",
        "cortical",
        "dorsolateral prefrontal cortex",
    },
    "liver": {"hepatic", "HepG2", "right lobe of liver"},
    "kidney": {"renal", "nephric"},
    "heart": {"cardiac", "myocardial", "heart left ventricle", "heart right ventricle"},
    "lung": {"pulmonary", "respiratory", "A549"},
    "blood": {"hematopoietic", "K562", "GM12878"},
    "colon": {
        "large intestine",
        "colonic",
        "transverse colon",
        "sigmoid colon",
        "HCT116",
    },
    "muscle": {"muscular", "myogenic", "gastrocnemius"},
}


# =============================================================================
# LEGACY COMPATIBILITY EXPORTS
# These provide backward compatibility with old code that imported
# specific dictionaries. They are now computed from JSON data.
# =============================================================================


def _build_assay_types_dict() -> dict[str, str]:
    """Build ASSAY_TYPES dict for backward compatibility.

    Returns dict mapping assay_name -> display string with count.
    """
    result = {}
    for name, count in get_assay_types():
        display = get_assay_display_name(name)
        result[name] = f"{display} ({count:,} experiments)"
    return result


def _build_top_biosamples_list(limit: int = 50) -> list[str]:
    """Build TOP_BIOSAMPLES list for backward compatibility."""
    return get_biosample_names(limit=limit)


def _build_top_targets_list(limit: int = 40) -> list[str]:
    """Build TOP_TARGETS list for backward compatibility."""
    return get_target_names(limit=limit)


def _build_life_stages_list() -> list[str]:
    """Build LIFE_STAGES list for backward compatibility."""
    return get_life_stage_names()


def _build_common_labs_list(limit: int = 20) -> list[str]:
    """Build COMMON_LABS list for backward compatibility."""
    return get_lab_names(limit=limit)


# Legacy exports - computed on first access
# These are properties that lazily compute from JSON
class _LazyDict(dict):
    """Dict that populates itself on first access."""

    def __init__(self, builder):
        self._builder = builder
        self._built = False

    def _ensure_built(self):
        if not self._built:
            self.update(self._builder())
            self._built = True

    def __getitem__(self, key):
        self._ensure_built()
        return super().__getitem__(key)

    def __contains__(self, key):
        self._ensure_built()
        return super().__contains__(key)

    def __iter__(self):
        self._ensure_built()
        return super().__iter__()

    def keys(self):
        self._ensure_built()
        return super().keys()

    def values(self):
        self._ensure_built()
        return super().values()

    def items(self):
        self._ensure_built()
        return super().items()

    def __len__(self):
        self._ensure_built()
        return super().__len__()

    def get(self, key, default=None):
        self._ensure_built()
        return super().get(key, default)


class _LazyList(list):
    """List that populates itself on first access."""

    def __init__(self, builder):
        self._builder = builder
        self._built = False

    def _ensure_built(self):
        if not self._built:
            self.extend(self._builder())
            self._built = True

    def __getitem__(self, index):
        self._ensure_built()
        return super().__getitem__(index)

    def __iter__(self):
        self._ensure_built()
        return super().__iter__()

    def __len__(self):
        self._ensure_built()
        return super().__len__()

    def __contains__(self, item):
        self._ensure_built()
        return super().__contains__(item)


# Legacy exports for backward compatibility
ASSAY_TYPES = _LazyDict(_build_assay_types_dict)
TOP_BIOSAMPLES = _LazyList(lambda: _build_top_biosamples_list(50))
TOP_TARGETS = _LazyList(lambda: _build_top_targets_list(40))
LIFE_STAGES = _LazyList(_build_life_stages_list)
COMMON_LABS = _LazyList(lambda: _build_common_labs_list(20))


# =============================================================================
# BACKWARD COMPATIBLE HELPER FUNCTIONS
# =============================================================================


def get_all_assay_types() -> list[str]:
    """Return list of all assay type keys, ordered by experiment count.

    For new code, prefer get_assay_types() which returns (name, count) tuples.
    """
    return get_assay_type_names()


def get_all_developmental_stages() -> list[str]:
    """Return list of all life stage keys.

    Note: This returns ACTUAL ENCODE life stages (like "adult", "embryonic"),
    NOT fabricated developmental stages like "E10.5" or "P56".
    """
    return get_life_stage_names()


def get_top_biosamples(limit: int = 50) -> list[str]:
    """Return top biosamples by experiment count."""
    return get_biosample_names(limit=limit)


def get_top_targets(limit: int = 20) -> list[str]:
    """Return top targets by experiment count."""
    return get_target_names(limit=limit)
