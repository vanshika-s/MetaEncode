# scripts/fetch_encode_facets.py
"""Fetch all valid vocabulary values from ENCODE API.

This script queries the ENCODE API to get ALL valid values for search filters
directly from the source of truth. Values are counted from actual experiments,
allowing us to order by popularity (most experiments first).

Strategy:
- Query all experiments with minimal fields to minimize response size
- Extract and count unique values for each field we need
- Use the actual field names that work with ENCODE's search API

Output:
- encode_facets_raw.json: Raw extracted data for verification
- generated_vocabularies.py: Python code ready to update vocabularies.py
"""

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import requests

# ENCODE API configuration
BASE_URL = "https://www.encodeproject.org"
HEADERS = {"accept": "application/json"}

# Fields to extract from experiments (these are the API filter parameter names)
FIELDS_TO_EXTRACT = [
    "assay_term_name",  # Main assay type filter
    "biosample_ontology.term_name",  # Biosample filter
    "target.label",  # ChIP-seq target filter
    "lab.title",  # Lab filter
    "status",  # Status filter
]

# These fields need to be extracted from nested structures
NESTED_FIELDS = {
    "organism": "replicates.library.biosample.donor.organism.scientific_name",
    "life_stage": "replicates.library.biosample.life_stage",
}


def fetch_all_experiments_minimal() -> list[dict]:
    """Fetch all experiments with minimal fields for efficiency.

    Returns:
        List of experiment records with only needed fields
    """
    # Request only the fields we need to minimize response size
    fields = [
        "accession",
        "assay_term_name",
        "biosample_ontology.term_name",
        "biosample_ontology.organ_slims",
        "biosample_ontology.cell_slims",
        "biosample_ontology.developmental_slims",
        "biosample_ontology.system_slims",
        "target.label",
        "lab.title",
        "status",
        "replicates.library.biosample.donor.organism.scientific_name",
        "replicates.library.biosample.life_stage",
    ]

    params = {
        "type": "Experiment",
        "format": "json",
        "limit": "all",
    }

    # Add field parameters
    field_params = "&".join(f"field={f}" for f in fields)
    url = f"{BASE_URL}/search/?{urlencode(params)}&{field_params}"

    print(f"Fetching all experiments from ENCODE API...")
    print(f"URL: {url[:100]}...")

    response = requests.get(url, headers=HEADERS, timeout=300)
    response.raise_for_status()

    data = response.json()
    return data.get("@graph", [])


def extract_nested_value(experiment: dict, field_path: str) -> str | None:
    """Extract value from a nested field path like 'replicates.library.biosample.donor.organism.scientific_name'.

    Args:
        experiment: Experiment record
        field_path: Dot-separated path to the field

    Returns:
        First non-empty value found, or None
    """
    parts = field_path.split(".")

    def traverse(obj, remaining_parts):
        if not remaining_parts:
            return obj if isinstance(obj, str) else None

        current_part = remaining_parts[0]
        rest = remaining_parts[1:]

        if isinstance(obj, dict):
            return traverse(obj.get(current_part), rest)
        elif isinstance(obj, list):
            # For lists, try to get value from first element
            for item in obj:
                result = traverse(item, [current_part] + rest if current_part else rest)
                if result:
                    return result
            return None
        return None

    return traverse(experiment, parts)


def count_field_values(experiments: list[dict]) -> dict[str, Counter]:
    """Count unique values for each field across all experiments.

    Args:
        experiments: List of experiment records

    Returns:
        Dictionary mapping field names to Counter objects
    """
    counters = {
        "assay_term_name": Counter(),
        "biosample_ontology.term_name": Counter(),
        "target.label": Counter(),
        "lab.title": Counter(),
        "status": Counter(),
        "organism": Counter(),
        "life_stage": Counter(),
    }

    for exp in experiments:
        # Simple fields
        if assay := exp.get("assay_term_name"):
            counters["assay_term_name"][assay] += 1

        # Nested field: biosample_ontology.term_name
        biosample_onto = exp.get("biosample_ontology", {})
        if isinstance(biosample_onto, dict):
            if term := biosample_onto.get("term_name"):
                counters["biosample_ontology.term_name"][term] += 1

        # Nested field: target.label
        target = exp.get("target", {})
        if isinstance(target, dict):
            if label := target.get("label"):
                counters["target.label"][label] += 1

        # Nested field: lab.title
        lab = exp.get("lab", {})
        if isinstance(lab, dict):
            if title := lab.get("title"):
                counters["lab.title"][title] += 1

        # Status
        if status := exp.get("status"):
            counters["status"][status] += 1

        # Extract from replicates for organism and life_stage
        replicates = exp.get("replicates", [])
        if replicates and isinstance(replicates, list):
            for rep in replicates:
                if not isinstance(rep, dict):
                    continue
                library = rep.get("library", {})
                if not isinstance(library, dict):
                    continue
                biosample = library.get("biosample", {})
                if not isinstance(biosample, dict):
                    continue

                # Life stage
                if life_stage := biosample.get("life_stage"):
                    counters["life_stage"][life_stage] += 1

                # Organism from donor
                donor = biosample.get("donor", {})
                if isinstance(donor, dict):
                    organism = donor.get("organism", {})
                    if isinstance(organism, dict):
                        if sci_name := organism.get("scientific_name"):
                            counters["organism"][sci_name] += 1

    return counters


def build_slim_to_biosamples(
    experiments: list[dict], slim_field: str
) -> dict[str, Counter]:
    """Build mapping of slim_value -> biosample term_names with counts.

    Generic function for all slim types (organ_slims, cell_slims,
    developmental_slims, system_slims).

    Args:
        experiments: List of experiment records with biosample_ontology.
        slim_field: The slim field name (e.g., "organ_slims", "cell_slims").

    Returns:
        Dictionary mapping slim values to Counter of biosample term_names.
    """
    slim_to_biosamples: dict[str, Counter] = defaultdict(Counter)

    for exp in experiments:
        biosample_onto = exp.get("biosample_ontology", {})
        if not isinstance(biosample_onto, dict):
            continue

        term_name = biosample_onto.get("term_name")
        slim_values = biosample_onto.get(slim_field, [])

        if term_name and slim_values and isinstance(slim_values, list):
            for slim_value in slim_values:
                slim_to_biosamples[slim_value][term_name] += 1

    return slim_to_biosamples


def build_organ_to_biosamples(experiments: list[dict]) -> dict[str, Counter]:
    """Build mapping of organ_slim -> biosample term_names with counts.

    This uses ENCODE's organ_slims from biosample_ontology, which are
    UBERON ontology-derived standardized organ classifications.

    Wrapper for backwards compatibility. Uses generic build_slim_to_biosamples.

    Args:
        experiments: List of experiment records with biosample_ontology.

    Returns:
        Dictionary mapping organ names to Counter of biosample term_names.
    """
    return build_slim_to_biosamples(experiments, "organ_slims")


def generate_vocabularies_code(counters: dict[str, Counter]) -> str:
    """Generate Python code for vocabularies.py from counted values.

    Args:
        counters: Dictionary of field -> Counter objects

    Returns:
        Python code string
    """
    lines = []
    lines.append("# Auto-generated from ENCODE API experiment data")
    lines.append(f"# Generated: {datetime.now().isoformat()}")
    lines.append("# Source: ENCODE API /search/?type=Experiment (all experiments)")
    lines.append("# Values are ordered by experiment count (most popular first)")
    lines.append("")

    # Assay types
    if "assay_term_name" in counters and counters["assay_term_name"]:
        lines.append(
            "# ============================================================================="
        )
        lines.append("# ASSAY TYPES (from ENCODE API, ordered by experiment count)")
        lines.append("# Filter parameter: assay_term_name")
        lines.append(
            "# ============================================================================="
        )
        lines.append("ASSAY_TYPES_FROM_ENCODE: list[tuple[str, int]] = [")
        for key, count in counters["assay_term_name"].most_common():
            lines.append(f'    ("{key}", {count}),')
        lines.append("]")
        lines.append("")

    # Biosamples
    if (
        "biosample_ontology.term_name" in counters
        and counters["biosample_ontology.term_name"]
    ):
        lines.append(
            "# ============================================================================="
        )
        lines.append("# BIOSAMPLES (from ENCODE API, ordered by experiment count)")
        lines.append("# Filter parameter: biosample_ontology.term_name")
        lines.append(
            "# ============================================================================="
        )
        lines.append("BIOSAMPLES_FROM_ENCODE: list[tuple[str, int]] = [")
        for key, count in counters["biosample_ontology.term_name"].most_common():
            key_escaped = key.replace('"', '\\"').replace("'", "\\'")
            lines.append(f'    ("{key_escaped}", {count}),')
        lines.append("]")
        lines.append("")

    # Organisms
    if "organism" in counters and counters["organism"]:
        lines.append(
            "# ============================================================================="
        )
        lines.append("# ORGANISMS (from ENCODE API, ordered by experiment count)")
        lines.append(
            "# Filter parameter: replicates.library.biosample.donor.organism.scientific_name"
        )
        lines.append(
            "# ============================================================================="
        )
        lines.append("ORGANISMS_FROM_ENCODE: list[tuple[str, int]] = [")
        for key, count in counters["organism"].most_common():
            lines.append(f'    ("{key}", {count}),')
        lines.append("]")
        lines.append("")

    # Targets (for ChIP-seq)
    if "target.label" in counters and counters["target.label"]:
        lines.append(
            "# ============================================================================="
        )
        lines.append(
            "# TARGETS - ChIP-seq targets (from ENCODE API, ordered by experiment count)"
        )
        lines.append("# Filter parameter: target.label")
        lines.append(
            "# ============================================================================="
        )
        lines.append("TARGETS_FROM_ENCODE: list[tuple[str, int]] = [")
        for key, count in counters["target.label"].most_common():
            key_escaped = key.replace('"', '\\"').replace("'", "\\'")
            lines.append(f'    ("{key_escaped}", {count}),')
        lines.append("]")
        lines.append("")

    # Life stages
    if "life_stage" in counters and counters["life_stage"]:
        lines.append(
            "# ============================================================================="
        )
        lines.append("# LIFE STAGES (from ENCODE API, ordered by experiment count)")
        lines.append("# Filter parameter: replicates.library.biosample.life_stage")
        lines.append(
            "# ============================================================================="
        )
        lines.append("LIFE_STAGES_FROM_ENCODE: list[tuple[str, int]] = [")
        for key, count in counters["life_stage"].most_common():
            lines.append(f'    ("{key}", {count}),')
        lines.append("]")
        lines.append("")

    # Labs
    if "lab.title" in counters and counters["lab.title"]:
        lines.append(
            "# ============================================================================="
        )
        lines.append("# LABS (from ENCODE API, ordered by experiment count)")
        lines.append("# Filter parameter: lab.title")
        lines.append(
            "# ============================================================================="
        )
        lines.append("LABS_FROM_ENCODE: list[tuple[str, int]] = [")
        for key, count in counters["lab.title"].most_common():
            key_escaped = key.replace('"', '\\"').replace("'", "\\'")
            lines.append(f'    ("{key_escaped}", {count}),')
        lines.append("]")
        lines.append("")

    # Status values
    if "status" in counters and counters["status"]:
        lines.append(
            "# ============================================================================="
        )
        lines.append("# STATUS VALUES (from ENCODE API)")
        lines.append("# Filter parameter: status")
        lines.append(
            "# ============================================================================="
        )
        lines.append("STATUS_VALUES_FROM_ENCODE: list[tuple[str, int]] = [")
        for key, count in counters["status"].most_common():
            lines.append(f'    ("{key}", {count}),')
        lines.append("]")
        lines.append("")

    return "\n".join(lines)


def main():
    """Main entry point."""
    print("=" * 70)
    print("ENCODE API Vocabulary Fetcher")
    print("=" * 70)
    print()

    # Fetch all experiments from ENCODE API
    try:
        experiments = fetch_all_experiments_minimal()
    except requests.RequestException as e:
        print(f"ERROR: Failed to fetch from ENCODE API: {e}")
        sys.exit(1)

    total_experiments = len(experiments)
    print(f"Total experiments fetched: {total_experiments}")
    print()

    # Count field values
    print("Counting unique values for each field...")
    counters = count_field_values(experiments)

    # Build all slim -> biosample mappings
    print("Building slim to biosample mappings...")
    organ_mapping = build_slim_to_biosamples(experiments, "organ_slims")
    cell_mapping = build_slim_to_biosamples(experiments, "cell_slims")
    developmental_mapping = build_slim_to_biosamples(experiments, "developmental_slims")
    system_mapping = build_slim_to_biosamples(experiments, "system_slims")

    # Print summary
    print()
    print("Field value summary:")
    print("-" * 50)
    for field, counter in counters.items():
        print(f"  {field}: {len(counter)} unique values")
    print(f"  organ_slims: {len(organ_mapping)} unique organs")
    print(f"  cell_slims: {len(cell_mapping)} unique cell types")
    print(f"  developmental_slims: {len(developmental_mapping)} unique germ layers")
    print(f"  system_slims: {len(system_mapping)} unique body systems")
    print()

    # Save raw data for verification
    data_dir = Path(__file__).parent.parent / "data"
    raw_output_path = data_dir / "encode_facets_raw.json"

    # Convert Counters to serializable format
    raw_data = {
        "fetched_at": datetime.now().isoformat(),
        "total_experiments": total_experiments,
        "field_counts": {
            field: [{"key": k, "count": v} for k, v in counter.most_common()]
            for field, counter in counters.items()
        },
        "organ_to_biosamples": {
            organ: [{"key": k, "count": v} for k, v in counter.most_common()]
            for organ, counter in sorted(organ_mapping.items())
        },
        "cell_to_biosamples": {
            cell: [{"key": k, "count": v} for k, v in counter.most_common()]
            for cell, counter in sorted(cell_mapping.items())
        },
        "developmental_to_biosamples": {
            dev: [{"key": k, "count": v} for k, v in counter.most_common()]
            for dev, counter in sorted(developmental_mapping.items())
        },
        "system_to_biosamples": {
            system: [{"key": k, "count": v} for k, v in counter.most_common()]
            for system, counter in sorted(system_mapping.items())
        },
    }

    with open(raw_output_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"Saved raw data to: {raw_output_path}")

    # Generate Python code (legacy output, can be removed - app uses JSON directly)
    generated_code = generate_vocabularies_code(counters)
    generated_output_path = data_dir.parent / "scripts" / "generated_vocabularies.py"

    with open(generated_output_path, "w") as f:
        f.write(generated_code)
    print(f"Saved generated vocabularies to: {generated_output_path}")

    # Print highlights
    print()
    print("=" * 70)
    print("TOP VALUES BY CATEGORY")
    print("=" * 70)

    if counters["assay_term_name"]:
        print("\nTop 15 Assay Types:")
        for i, (key, count) in enumerate(
            counters["assay_term_name"].most_common(15), 1
        ):
            print(f"  {i:2}. {key}: {count} experiments")

    if counters["biosample_ontology.term_name"]:
        print("\nTop 15 Biosamples:")
        for i, (key, count) in enumerate(
            counters["biosample_ontology.term_name"].most_common(15), 1
        ):
            print(f"  {i:2}. {key}: {count} experiments")

    if counters["organism"]:
        print("\nOrganisms:")
        for key, count in counters["organism"].most_common():
            print(f"  - {key}: {count} experiments")

    if counters["target.label"]:
        print("\nTop 15 Targets (ChIP-seq):")
        for i, (key, count) in enumerate(counters["target.label"].most_common(15), 1):
            print(f"  {i:2}. {key}: {count} experiments")

    if counters["life_stage"]:
        print("\nLife Stages:")
        for key, count in counters["life_stage"].most_common():
            print(f"  - {key}: {count} experiments")

    # Helper to print slim mapping summary
    def print_slim_summary(name: str, mapping: dict[str, Counter], limit: int = 15):
        if mapping:
            print(f"\n{name}:")
            totals = [(key, sum(counter.values())) for key, counter in mapping.items()]
            totals.sort(key=lambda x: -x[1])
            for i, (key, total) in enumerate(totals[:limit], 1):
                biosample_count = len(mapping[key])
                print(
                    f"  {i:2}. {key}: {total} experiments ({biosample_count} biosamples)"
                )

    print_slim_summary("Organ Systems (organ_slims)", organ_mapping)
    print_slim_summary("Cell Types (cell_slims)", cell_mapping)
    print_slim_summary(
        "Developmental Layers (developmental_slims)", developmental_mapping
    )
    print_slim_summary("Body Systems (system_slims)", system_mapping)

    print()
    print("=" * 70)
    print("DONE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Review encode_facets_raw.json for verification")
    print("2. All four slim types are now available in the JSON")
    print("3. Use vocabularies.py accessor functions to access the data")


if __name__ == "__main__":
    main()
