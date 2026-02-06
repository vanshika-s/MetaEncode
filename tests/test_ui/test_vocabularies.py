# tests/test_ui/test_vocabularies.py
"""Tests for vocabulary definitions and helper functions.

Tests the new dynamic JSON loading architecture where all vocabulary
values are loaded from scripts/encode_facets_raw.json.
"""

from src.ui.vocabularies import (
    ASSAY_ALIASES,
    ASSAY_TYPES,
    BODY_PARTS,
    CELL_TYPE_DISPLAY_NAMES,
    COMMON_LABS,
    DEVELOPMENTAL_DISPLAY_NAMES,
    HISTONE_ALIASES,
    HISTONE_MODIFICATIONS,
    LIFE_STAGES,
    ORGAN_DISPLAY_NAMES,
    ORGANISM_ASSEMBLIES,
    ORGANISMS,
    SLIM_TYPES,
    SYSTEM_DISPLAY_NAMES,
    TISSUE_SYNONYMS,
    TOP_BIOSAMPLES,
    TOP_TARGETS,
    build_biosample_to_body_systems,
    build_biosample_to_cell_types,
    build_biosample_to_developmental_layers,
    build_biosample_to_organs,
    build_biosample_to_slim,
    format_assay_with_count,
    get_all_assay_types,
    get_all_body_parts,
    get_all_developmental_stages,
    get_all_histone_mods,
    get_all_organisms,
    get_all_organs_for_biosample,
    get_all_slims_for_biosample,
    get_assay_display_name,
    get_assay_types,
    get_biosample_names_for_organ,
    get_biosamples,
    get_biosamples_for_body_system,
    get_biosamples_for_cell_type,
    get_biosamples_for_developmental_layer,
    get_biosamples_for_organ,
    get_biosamples_for_slim,
    get_body_part_display_name,
    get_body_system_names,
    get_body_systems,
    get_cell_type_display_name,
    get_cell_type_names,
    get_cell_types,
    get_developmental_display_name,
    get_developmental_layer_names,
    get_developmental_layers,
    get_facets_timestamp,
    get_labs,
    get_life_stages,
    get_organ_display_name,
    get_organ_system_names,
    get_organ_systems,
    get_organism_common_name,
    get_organism_display,
    get_organism_names,
    get_organism_scientific_name,
    get_organisms,
    get_primary_body_system_for_biosample,
    get_primary_cell_type_for_biosample,
    get_primary_developmental_layer_for_biosample,
    get_primary_organ_for_biosample,
    get_primary_slim_for_biosample,
    get_slim_categories,
    get_slim_category_names,
    get_slim_display_name,
    get_system_display_name,
    get_target_description,
    get_targets,
    get_tissues_for_body_part,
    get_top_biosamples,
    get_top_targets,
    get_total_experiments,
    normalize_search_term,
    reload_facets,
)


class TestJSONLoading:
    """Tests for dynamic JSON loading functionality."""

    def test_json_loads_successfully(self) -> None:
        """Test that JSON data loads without errors."""
        # This implicitly tests _load_facets()
        total = get_total_experiments()
        assert total > 0

    def test_assay_types_loaded_from_json(self) -> None:
        """Test that assay types are loaded from JSON."""
        assays = get_assay_types()
        assert isinstance(assays, list)
        assert len(assays) > 0
        # Check structure: list of (name, count) tuples
        name, count = assays[0]
        assert isinstance(name, str)
        assert isinstance(count, int)
        assert count > 0

    def test_assay_types_ordered_by_popularity(self) -> None:
        """Test that assay types are ordered by experiment count (descending)."""
        assays = get_assay_types()
        counts = [count for name, count in assays]
        # Counts should be in descending order (most popular first)
        assert counts == sorted(counts, reverse=True)

    def test_chip_seq_is_first(self) -> None:
        """Test that ChIP-seq is the most popular assay type."""
        assays = get_assay_types()
        first_assay = assays[0][0]
        assert first_assay == "ChIP-seq"

    def test_biosamples_loaded_from_json(self) -> None:
        """Test that biosamples are loaded from JSON."""
        biosamples = get_biosamples()
        assert isinstance(biosamples, list)
        assert len(biosamples) > 100  # ENCODE has many biosamples

    def test_targets_loaded_from_json(self) -> None:
        """Test that targets are loaded from JSON."""
        targets = get_targets()
        assert isinstance(targets, list)
        assert len(targets) > 50  # ENCODE has many targets
        # H3K4me3 should be in top targets
        target_names = [name for name, count in targets[:20]]
        assert "H3K4me3" in target_names

    def test_life_stages_loaded_from_json(self) -> None:
        """Test that life stages are loaded from JSON."""
        stages = get_life_stages()
        assert isinstance(stages, list)
        assert len(stages) > 0
        # Check for actual ENCODE life stages
        stage_names = [name for name, count in stages]
        assert "adult" in stage_names
        assert "embryonic" in stage_names

    def test_life_stages_not_fabricated(self) -> None:
        """Test that life stages are real ENCODE values, not fabricated."""
        stages = get_life_stages()
        stage_names = [name for name, count in stages]
        # These fabricated values should NOT be present
        fabricated_stages = ["E10.5", "E14.5", "P0", "P56", "P60"]
        for fake_stage in fabricated_stages:
            assert fake_stage not in stage_names, f"Fabricated stage {fake_stage} found"

    def test_labs_loaded_from_json(self) -> None:
        """Test that labs are loaded from JSON."""
        labs = get_labs()
        assert isinstance(labs, list)
        assert len(labs) > 10  # ENCODE has many labs


class TestAssayTypes:
    """Tests for ASSAY_TYPES dictionary (legacy compatibility)."""

    def test_assay_types_not_empty(self) -> None:
        """Test that ASSAY_TYPES contains entries."""
        assert len(ASSAY_TYPES) > 0

    def test_assay_types_contains_common_assays(self) -> None:
        """Test that common ENCODE assays are present."""
        # Note: ENCODE uses "HiC" not "Hi-C" as the canonical spelling
        common_assays = ["ChIP-seq", "RNA-seq", "ATAC-seq", "DNase-seq", "HiC"]
        for assay in common_assays:
            assert assay in ASSAY_TYPES

    def test_hic_variants_present(self) -> None:
        """Test that HiC-related assays are present."""
        # Note: ENCODE uses "HiC" (not "Hi-C") and "capture Hi-C" (not "in situ Hi-C")
        hic_variants = ["HiC", "capture Hi-C"]
        for variant in hic_variants:
            assert variant in ASSAY_TYPES

    def test_assay_types_have_display_names(self) -> None:
        """Test that all assay types have non-empty display names."""
        for key, display in ASSAY_TYPES.items():
            assert display, f"Empty display name for {key}"
            assert isinstance(display, str)

    def test_assay_aliases_reference_valid_assays(self) -> None:
        """Test that all aliases reference valid assay types."""
        for assay_key in ASSAY_ALIASES.keys():
            assert assay_key in ASSAY_TYPES, f"Alias key {assay_key} not in ASSAY_TYPES"

    def test_assay_aliases_are_lists(self) -> None:
        """Test that all alias values are non-empty lists of strings."""
        for key, aliases in ASSAY_ALIASES.items():
            assert isinstance(aliases, list), f"Aliases for {key} should be a list"
            assert len(aliases) > 0, f"Aliases for {key} should not be empty"
            for alias in aliases:
                assert isinstance(alias, str), f"Alias {alias} should be a string"


class TestDisplayNames:
    """Tests for display name functionality."""

    def test_get_assay_display_name_returns_short_name(self) -> None:
        """Test that long assay names get shortened."""
        long_name = "single-cell RNA sequencing assay"
        display = get_assay_display_name(long_name)
        assert display == "scRNA-seq"

    def test_get_assay_display_name_returns_original_for_unknown(self) -> None:
        """Test that unknown assays return original name."""
        unknown_assay = "ChIP-seq"
        display = get_assay_display_name(unknown_assay)
        assert display == "ChIP-seq"


class TestOrganisms:
    """Tests for ORGANISMS dictionary."""

    def test_organisms_not_empty(self) -> None:
        """Test that ORGANISMS contains entries."""
        assert len(ORGANISMS) > 0

    def test_organisms_contains_human_and_mouse(self) -> None:
        """Test that human and mouse are present."""
        assert "human" in ORGANISMS
        assert "mouse" in ORGANISMS

    def test_organisms_have_required_fields(self) -> None:
        """Test that all organisms have required fields."""
        required_fields = ["display_name", "scientific_name", "assembly"]
        for org_key, org_info in ORGANISMS.items():
            for field in required_fields:
                assert field in org_info, f"Missing {field} for {org_key}"

    def test_human_has_correct_assembly(self) -> None:
        """Test that human has hg38 assembly."""
        assert ORGANISMS["human"]["assembly"] == "hg38"

    def test_mouse_has_correct_assembly(self) -> None:
        """Test that mouse has mm10 assembly."""
        assert ORGANISMS["mouse"]["assembly"] == "mm10"


class TestDynamicOrganisms:
    """Tests for dynamic organism loading from ENCODE JSON."""

    def test_get_organisms_returns_list_of_tuples(self) -> None:
        """Test that get_organisms returns list of (name, count) tuples."""
        result = get_organisms()
        assert isinstance(result, list)
        assert len(result) > 0
        # Each item should be (scientific_name, count)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], int)

    def test_get_organisms_includes_main_model_organisms(self) -> None:
        """Test that main model organisms are in the dynamic list."""
        organisms = get_organisms()
        sci_names = [name for name, _ in organisms]
        assert "Homo sapiens" in sci_names
        assert "Mus musculus" in sci_names
        assert "Drosophila melanogaster" in sci_names
        assert "Caenorhabditis elegans" in sci_names

    def test_get_organisms_includes_minor_species(self) -> None:
        """Test that minor species from ENCODE are included."""
        organisms = get_organisms()
        sci_names = [name for name, _ in organisms]
        # There should be more than the 4 main model organisms
        assert len(sci_names) > 4

    def test_get_organism_names_returns_strings(self) -> None:
        """Test that get_organism_names returns list of strings."""
        result = get_organism_names()
        assert isinstance(result, list)
        assert all(isinstance(name, str) for name in result)
        assert "Homo sapiens" in result

    def test_get_organism_names_with_limit(self) -> None:
        """Test that limit parameter works."""
        result = get_organism_names(limit=3)
        assert len(result) == 3

    def test_organism_assemblies_structure(self) -> None:
        """Test ORGANISM_ASSEMBLIES has correct structure."""
        assert "Homo sapiens" in ORGANISM_ASSEMBLIES
        assert "Mus musculus" in ORGANISM_ASSEMBLIES

        human = ORGANISM_ASSEMBLIES["Homo sapiens"]
        assert human["common_name"] == "human"
        assert human["short_name"] == "Human"
        assert human["assembly"] == "hg38"

    def test_get_organism_common_name(self) -> None:
        """Test get_organism_common_name function."""
        assert get_organism_common_name("Homo sapiens") == "human"
        assert get_organism_common_name("Mus musculus") == "mouse"
        assert get_organism_common_name("Unknown species") is None

    def test_get_organism_scientific_name(self) -> None:
        """Test get_organism_scientific_name function."""
        # From common name
        assert get_organism_scientific_name("human") == "Homo sapiens"
        assert get_organism_scientific_name("mouse") == "Mus musculus"
        # Scientific name passes through
        assert get_organism_scientific_name("Homo sapiens") == "Homo sapiens"
        # Unknown passes through
        assert get_organism_scientific_name("unknown") == "unknown"


class TestHistoneModifications:
    """Tests for HISTONE_MODIFICATIONS dictionary."""

    def test_histone_mods_not_empty(self) -> None:
        """Test that HISTONE_MODIFICATIONS contains entries."""
        assert len(HISTONE_MODIFICATIONS) > 0

    def test_common_histone_marks_present(self) -> None:
        """Test that common histone marks are present."""
        common_marks = ["H3K27ac", "H3K4me3", "H3K4me1", "H3K27me3", "CTCF"]
        for mark in common_marks:
            assert mark in HISTONE_MODIFICATIONS

    def test_histone_mods_have_required_fields(self) -> None:
        """Test that all histone mods have required fields."""
        required_fields = ["full_name", "description", "category"]
        for mark_key, mark_info in HISTONE_MODIFICATIONS.items():
            for field in required_fields:
                assert field in mark_info, f"Missing {field} for {mark_key}"

    def test_histone_categories_are_valid(self) -> None:
        """Test that histone modification categories are valid."""
        valid_categories = {
            "active",
            "enhancer",
            "promoter",
            "transcription",
            "repressive",
            "tf",
        }
        for mark_key, mark_info in HISTONE_MODIFICATIONS.items():
            assert (
                mark_info["category"] in valid_categories
            ), f"Invalid category for {mark_key}"

    def test_histone_aliases_reference_valid_marks(self) -> None:
        """Test that all histone aliases reference valid modifications."""
        for mark_key in HISTONE_ALIASES.keys():
            assert (
                mark_key in HISTONE_MODIFICATIONS
            ), f"Alias key {mark_key} not in HISTONE_MODIFICATIONS"


class TestBodyParts:
    """Tests for BODY_PARTS dictionary."""

    def test_body_parts_not_empty(self) -> None:
        """Test that BODY_PARTS contains entries."""
        assert len(BODY_PARTS) > 0

    def test_common_body_parts_present(self) -> None:
        """Test that common body parts are present."""
        common_parts = ["brain", "heart", "liver", "kidney", "lung", "blood"]
        for part in common_parts:
            assert part in BODY_PARTS

    def test_body_parts_have_required_fields(self) -> None:
        """Test that all body parts have required fields."""
        for part_key, part_info in BODY_PARTS.items():
            assert "display_name" in part_info, f"Missing display_name for {part_key}"
            assert "tissues" in part_info, f"Missing tissues for {part_key}"
            assert isinstance(
                part_info["tissues"], list
            ), f"Tissues for {part_key} should be a list"
            assert (
                len(part_info["tissues"]) > 0
            ), f"Tissues for {part_key} should not be empty"

    def test_brain_contains_cerebellum(self) -> None:
        """Test that brain body part includes cerebellum."""
        assert "cerebellum" in BODY_PARTS["brain"]["tissues"]

    def test_cell_line_body_part_contains_k562(self) -> None:
        """Test that cell_line body part includes K562."""
        assert "K562" in BODY_PARTS["cell_line"]["tissues"]

    def test_body_parts_have_aliases(self) -> None:
        """Test that most body parts have aliases."""
        # At least some body parts should have aliases
        parts_with_aliases = [
            key
            for key, info in BODY_PARTS.items()
            if "aliases" in info and len(info.get("aliases", [])) > 0
        ]
        assert len(parts_with_aliases) > 5, "Most body parts should have aliases"


class TestTissueSynonyms:
    """Tests for TISSUE_SYNONYMS dictionary."""

    def test_tissue_synonyms_not_empty(self) -> None:
        """Test that TISSUE_SYNONYMS contains entries."""
        assert len(TISSUE_SYNONYMS) > 0

    def test_cerebellum_hindbrain_synonyms(self) -> None:
        """Test that cerebellum and hindbrain are synonyms."""
        assert "hindbrain" in TISSUE_SYNONYMS["cerebellum"]
        assert "cerebellum" in TISSUE_SYNONYMS["hindbrain"]

    def test_synonyms_are_sets(self) -> None:
        """Test that all synonym values are sets."""
        for key, synonyms in TISSUE_SYNONYMS.items():
            assert isinstance(synonyms, set), f"Synonyms for {key} should be a set"


class TestLifeStages:
    """Tests for life stages (replacing DEVELOPMENTAL_STAGES)."""

    def test_life_stages_not_empty(self) -> None:
        """Test that LIFE_STAGES contains entries."""
        assert len(LIFE_STAGES) > 0

    def test_real_encode_stages_present(self) -> None:
        """Test that real ENCODE life stages are present."""
        real_stages = ["adult", "embryonic", "child", "newborn"]
        for stage in real_stages:
            assert stage in LIFE_STAGES, f"Real ENCODE stage {stage} not found"

    def test_fabricated_stages_not_present(self) -> None:
        """Test that fabricated developmental stages are NOT present."""
        # These were invented values that don't exist in ENCODE
        fabricated = ["E10.5", "E14.5", "P0", "P7", "P56", "P60", "8 weeks"]
        for stage in fabricated:
            assert (
                stage not in LIFE_STAGES
            ), f"Fabricated stage {stage} should not be in LIFE_STAGES"


class TestCommonLabs:
    """Tests for COMMON_LABS list."""

    def test_common_labs_not_empty(self) -> None:
        """Test that COMMON_LABS contains entries."""
        assert len(COMMON_LABS) > 0

    def test_common_labs_are_strings(self) -> None:
        """Test that all labs are non-empty strings."""
        for lab in COMMON_LABS:
            assert isinstance(lab, str)
            assert len(lab) > 0


class TestHelperFunctions:
    """Tests for vocabulary helper functions."""

    def test_get_all_assay_types_returns_list(self) -> None:
        """Test that get_all_assay_types returns a list ordered by popularity."""
        result = get_all_assay_types()
        assert isinstance(result, list)
        assert len(result) == len(ASSAY_TYPES)
        # Lists are ordered by experiment count (popularity), not alphabetically
        # ChIP-seq should be first as it has most experiments
        assert result[0] == "ChIP-seq"

    def test_get_all_organisms_returns_list(self) -> None:
        """Test that get_all_organisms returns scientific names from ENCODE."""
        result = get_all_organisms()
        assert isinstance(result, list)
        # Should include all organisms from ENCODE JSON (more than 4 main ones)
        assert len(result) >= 4
        # Returns scientific names, not common names
        assert "Homo sapiens" in result
        assert "Mus musculus" in result

    def test_get_organism_display_known_organism(self) -> None:
        """Test get_organism_display for known organisms."""
        # Test with common name
        human_display = get_organism_display("human")
        assert "Human" in human_display
        assert "hg38" in human_display

        # Test with scientific name
        human_display2 = get_organism_display("Homo sapiens")
        assert "Human" in human_display2
        assert "hg38" in human_display2

        mouse_display = get_organism_display("mouse")
        assert "Mouse" in mouse_display
        assert "mm10" in mouse_display

    def test_get_organism_display_unknown_organism(self) -> None:
        """Test get_organism_display for unknown organisms."""
        # Unknown organisms just return the input
        result = get_organism_display("unknown_organism")
        assert result == "unknown_organism"

        # Scientific names not in assembly dict return as-is
        result2 = get_organism_display("Drosophila simulans")
        assert result2 == "Drosophila simulans"

    def test_get_all_histone_mods_returns_list(self) -> None:
        """Test that get_all_histone_mods returns a list."""
        result = get_all_histone_mods()
        assert isinstance(result, list)
        assert len(result) == len(HISTONE_MODIFICATIONS)
        # Should contain common histone marks
        assert "H3K27ac" in result
        assert "H3K4me3" in result
        assert "CTCF" in result

    def test_get_all_body_parts_returns_list(self) -> None:
        """Test that get_all_body_parts returns a list."""
        result = get_all_body_parts()
        assert isinstance(result, list)
        assert len(result) == len(BODY_PARTS)
        assert "brain" in result

    def test_get_tissues_for_body_part_valid(self) -> None:
        """Test get_tissues_for_body_part for valid body part."""
        brain_tissues = get_tissues_for_body_part("brain")
        assert isinstance(brain_tissues, list)
        assert len(brain_tissues) > 0
        assert "cerebellum" in brain_tissues

    def test_get_tissues_for_body_part_invalid(self) -> None:
        """Test get_tissues_for_body_part for invalid body part."""
        result = get_tissues_for_body_part("invalid_body_part")
        assert result == []

    def test_get_all_developmental_stages_returns_list(self) -> None:
        """Test that get_all_developmental_stages returns actual ENCODE life stages."""
        result = get_all_developmental_stages()
        assert isinstance(result, list)
        assert len(result) > 0
        # Should contain real ENCODE stages
        assert "adult" in result


class TestVocabularyConsistency:
    """Tests for consistency across vocabularies."""

    def test_tissue_synonyms_are_strings(self) -> None:
        """Test that all synonym values are non-empty strings."""
        for key, synonyms in TISSUE_SYNONYMS.items():
            for syn in synonyms:
                assert isinstance(
                    syn, str
                ), f"Synonym '{syn}' for '{key}' should be a string"
                assert len(syn) > 0, f"Synonym for '{key}' should not be empty"

    def test_no_duplicate_tissues_in_body_part(self) -> None:
        """Test that there are no duplicate tissues within a body part."""
        for part_key, part_info in BODY_PARTS.items():
            tissues = part_info["tissues"]
            tissues_lower = [t.lower() for t in tissues]
            assert len(tissues_lower) == len(
                set(tissues_lower)
            ), f"Duplicate tissues in {part_key}"

    def test_top_biosamples_matches_json(self) -> None:
        """Test that TOP_BIOSAMPLES comes from JSON data."""
        top_biosamples = list(TOP_BIOSAMPLES)
        json_biosamples = get_biosamples()[:50]
        json_names = [name for name, count in json_biosamples]
        assert top_biosamples == json_names

    def test_top_targets_matches_json(self) -> None:
        """Test that TOP_TARGETS comes from JSON data."""
        top_targets = list(TOP_TARGETS)
        json_targets = get_targets()[:40]
        json_names = [name for name, count in json_targets]
        assert top_targets == json_names


class TestOrganSystems:
    """Tests for organ_slims-based functions."""

    def test_get_organ_systems_returns_list(self) -> None:
        """Test that get_organ_systems returns ordered list of tuples."""
        organs = get_organ_systems()
        assert isinstance(organs, list)
        assert len(organs) > 0
        # Should be (name, count) tuples
        name, count = organs[0]
        assert isinstance(name, str)
        assert isinstance(count, int)
        assert count > 0

    def test_get_organ_systems_ordered_by_count(self) -> None:
        """Test that organ systems are ordered by experiment count (descending)."""
        organs = get_organ_systems()
        counts = [count for name, count in organs]
        assert counts == sorted(counts, reverse=True)

    def test_get_organ_systems_contains_common_organs(self) -> None:
        """Test that common organs are present."""
        organs = get_organ_systems()
        organ_names = [name for name, count in organs]
        common_organs = ["brain", "heart", "liver", "lung", "kidney", "blood"]
        for organ in common_organs:
            assert organ in organ_names, f"Expected {organ} in organ systems"

    def test_get_organ_system_names(self) -> None:
        """Test that get_organ_system_names returns list of strings."""
        names = get_organ_system_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)
        # Should match the names from get_organ_systems
        full_data = get_organ_systems()
        expected_names = [name for name, count in full_data]
        assert names == expected_names

    def test_get_biosamples_for_organ_brain(self) -> None:
        """Test getting biosamples for brain organ."""
        biosamples = get_biosamples_for_organ("brain")
        assert isinstance(biosamples, list)
        # Brain should have multiple biosamples
        assert len(biosamples) > 5
        # Check structure
        name, count = biosamples[0]
        assert isinstance(name, str)
        assert isinstance(count, int)
        assert count > 0
        # Common brain tissues should be present
        biosample_names = [name for name, count in biosamples]
        assert any("cortex" in name.lower() for name in biosample_names)

    def test_get_biosamples_for_organ_ordered_by_count(self) -> None:
        """Test that biosamples for an organ are ordered by count."""
        biosamples = get_biosamples_for_organ("brain")
        counts = [count for name, count in biosamples]
        assert counts == sorted(counts, reverse=True)

    def test_get_biosamples_for_invalid_organ(self) -> None:
        """Test that invalid organ returns empty list."""
        biosamples = get_biosamples_for_organ("nonexistent_organ")
        assert biosamples == []

    def test_get_biosample_names_for_organ(self) -> None:
        """Test getting biosample names (without counts) for an organ."""
        names = get_biosample_names_for_organ("heart")
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)

    def test_get_biosample_names_for_organ_with_limit(self) -> None:
        """Test that limit parameter works correctly."""
        all_names = get_biosample_names_for_organ("brain")
        limited = get_biosample_names_for_organ("brain", limit=5)
        assert len(limited) == 5
        assert limited == all_names[:5]

    def test_get_organ_display_name_known(self) -> None:
        """Test display name for known organ mappings."""
        # "bodily fluid" -> "Blood / Bodily Fluid"
        assert get_organ_display_name("bodily fluid") == "Blood / Bodily Fluid"
        # "musculature of body" -> "Muscle"
        assert get_organ_display_name("musculature of body") == "Muscle"

    def test_get_organ_display_name_unknown(self) -> None:
        """Test display name for organs without custom mapping."""
        # Should title-case and replace underscores
        assert get_organ_display_name("brain") == "Brain"
        assert get_organ_display_name("test_organ") == "Test Organ"

    def test_organ_display_names_not_empty(self) -> None:
        """Test that ORGAN_DISPLAY_NAMES contains entries."""
        assert len(ORGAN_DISPLAY_NAMES) > 0

    def test_multiple_organs_have_biosamples(self) -> None:
        """Test that major organs all have biosample data."""
        organs_to_check = ["brain", "heart", "liver", "lung", "kidney"]
        for organ in organs_to_check:
            biosamples = get_biosamples_for_organ(organ)
            assert len(biosamples) > 0, f"Expected biosamples for {organ}"


class TestBiosampleToOrganMapping:
    """Tests for biosample-to-organ reverse mapping functions."""

    def test_build_biosample_to_organs_returns_dict(self) -> None:
        """Test that build_biosample_to_organs returns a non-empty dict."""
        mapping = build_biosample_to_organs()
        assert isinstance(mapping, dict)
        assert len(mapping) > 100  # Should have many biosamples mapped

    def test_build_biosample_to_organs_values_are_lists(self) -> None:
        """Test that mapping values are lists of organ names."""
        mapping = build_biosample_to_organs()
        for biosample, organs in list(mapping.items())[:10]:
            assert isinstance(organs, list)
            assert len(organs) > 0
            assert all(isinstance(org, str) for org in organs)

    def test_get_primary_organ_for_known_biosample(self) -> None:
        """Test getting primary organ for a known biosample."""
        # Cerebellum should map to brain
        organ = get_primary_organ_for_biosample("cerebellum")
        assert organ == "brain"

    def test_get_primary_organ_for_k562(self) -> None:
        """Test getting primary organ for K562 cell line."""
        organ = get_primary_organ_for_biosample("K562")
        assert organ == "blood"

    def test_get_primary_organ_for_unknown_biosample(self) -> None:
        """Test that unknown biosample returns None."""
        result = get_primary_organ_for_biosample("nonexistent_biosample_xyz")
        assert result is None

    def test_get_all_organs_for_biosample_single_organ(self) -> None:
        """Test biosample that maps to a single organ."""
        organs = get_all_organs_for_biosample("cerebellum")
        assert isinstance(organs, list)
        assert len(organs) >= 1
        assert "brain" in organs

    def test_get_all_organs_for_biosample_multiple_organs(self) -> None:
        """Test biosample that maps to multiple organs."""
        # Some biosamples map to multiple organs
        mapping = build_biosample_to_organs()
        # Find a biosample with multiple organs
        multi_organ_sample = None
        for biosample, organs in mapping.items():
            if len(organs) > 1:
                multi_organ_sample = biosample
                break
        assert (
            multi_organ_sample is not None
        ), "Should have at least one multi-organ biosample"
        organs = get_all_organs_for_biosample(multi_organ_sample)
        assert len(organs) > 1

    def test_get_all_organs_for_unknown_biosample(self) -> None:
        """Test that unknown biosample returns empty list."""
        organs = get_all_organs_for_biosample("nonexistent_sample")
        assert organs == []

    def test_organs_ordered_by_experiment_count(self) -> None:
        """Test that organs are ordered by experiment count (most popular first)."""
        # Get a biosample that maps to multiple organs
        mapping = build_biosample_to_organs()
        organ_counts = {name: count for name, count in get_organ_systems()}

        for biosample, organs in list(mapping.items())[:20]:
            if len(organs) > 1:
                # Check that organs are ordered by experiment count
                organ_experiment_counts = [organ_counts.get(org, 0) for org in organs]
                assert organ_experiment_counts == sorted(
                    organ_experiment_counts, reverse=True
                ), f"Organs for {biosample} not ordered by count"

    def test_all_biosamples_in_organ_mapping_can_be_reversed(self) -> None:
        """Test that biosamples from organ mapping appear in reverse mapping."""
        # Get some biosamples from an organ
        brain_biosamples = get_biosamples_for_organ("brain")[:10]
        mapping = build_biosample_to_organs()

        for name, _ in brain_biosamples:
            assert name in mapping, f"Biosample {name} should be in reverse mapping"
            assert "brain" in mapping[name], f"Brain should be in organs for {name}"


# =============================================================================
# Coverage Gap Tests
# =============================================================================


class TestUncoveredFunctions:
    """Tests for functions that need coverage."""

    def test_reload_facets(self) -> None:
        """Lines 63-64: reload_facets() forces cache refresh."""
        # First call to load facets
        initial_total = get_total_experiments()

        # Call reload_facets to force refresh
        reload_facets()

        # After reload, data should still be accessible
        new_total = get_total_experiments()
        assert new_total > 0
        assert new_total == initial_total  # Data should be same after reload

    def test_get_facets_timestamp(self) -> None:
        """Lines 177-178: get_facets_timestamp returns ISO timestamp string."""
        timestamp = get_facets_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0
        # Timestamp should contain date-like format or "unknown"
        assert "-" in timestamp or timestamp == "unknown"

    def test_format_assay_with_count(self) -> None:
        """Lines 484-485: format_assay_with_count returns formatted string."""
        result = format_assay_with_count("ChIP-seq", 12569)
        assert isinstance(result, str)
        assert "ChIP-seq" in result
        assert "12,569" in result  # Formatted with comma
        assert "experiments" in result

    def test_format_assay_with_count_long_name(self) -> None:
        """Test format_assay_with_count with a long assay name."""
        result = format_assay_with_count("single-cell RNA sequencing assay", 500)
        assert isinstance(result, str)
        # Should use display name (scRNA-seq)
        assert "scRNA-seq" in result
        assert "500" in result
        assert "experiments" in result

    def test_normalize_search_term_canonical(self) -> None:
        """Lines 533-539: normalize_search_term returns canonical when term matches."""
        result = normalize_search_term("ChIP-seq", ASSAY_ALIASES)
        assert result == "ChIP-seq"

    def test_normalize_search_term_alias_match(self) -> None:
        """Lines 533-539: normalize_search_term returns canonical for alias."""
        result = normalize_search_term("chip", ASSAY_ALIASES)
        assert result == "ChIP-seq"

    def test_normalize_search_term_no_match(self) -> None:
        """Lines 533-539: normalize_search_term returns None for unknown term."""
        result = normalize_search_term("unknown_assay", ASSAY_ALIASES)
        assert result is None

    def test_normalize_search_term_histone_alias(self) -> None:
        """Test normalize_search_term with histone aliases."""
        result = normalize_search_term("polycomb", HISTONE_ALIASES)
        assert result == "H3K27me3"

    def test_get_target_description_known(self) -> None:
        """Lines 775-777: get_target_description returns description for known target."""
        result = get_target_description("H3K27ac")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_target_description_unknown(self) -> None:
        """Lines 775-777: get_target_description returns None for unknown target."""
        result = get_target_description("UNKNOWN_TARGET_XYZ")
        assert result is None

    def test_get_target_description_ctcf(self) -> None:
        """Test get_target_description for CTCF."""
        result = get_target_description("CTCF")
        assert result is not None
        assert isinstance(result, str)

    def test_get_body_part_display_name_known(self) -> None:
        """Lines 988-991: get_body_part_display_name returns display name for known part."""
        result = get_body_part_display_name("brain")
        assert result == "Brain / Nervous System"

    def test_get_body_part_display_name_unknown(self) -> None:
        """Lines 988-991: get_body_part_display_name returns input for unknown part."""
        result = get_body_part_display_name("unknown_body_part")
        assert result == "unknown_body_part"

    def test_get_body_part_display_name_heart(self) -> None:
        """Test get_body_part_display_name for heart."""
        result = get_body_part_display_name("heart")
        assert "Heart" in result or "Cardiovascular" in result

    def test_get_top_biosamples(self) -> None:
        """Line 1171: get_top_biosamples returns list of biosample names."""
        result = get_top_biosamples()
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(name, str) for name in result)

    def test_get_top_biosamples_with_limit(self) -> None:
        """Test get_top_biosamples with custom limit."""
        result = get_top_biosamples(limit=10)
        assert len(result) == 10

    def test_get_top_targets(self) -> None:
        """Line 1176: get_top_targets returns list of target names."""
        result = get_top_targets()
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(name, str) for name in result)

    def test_get_top_targets_with_limit(self) -> None:
        """Test get_top_targets with custom limit."""
        result = get_top_targets(limit=5)
        assert len(result) == 5


class TestLazyCollections:
    """Tests for _LazyDict and _LazyList classes."""

    def test_lazy_dict_getitem(self) -> None:
        """Lines 1077-1079: _LazyDict.__getitem__ accesses items."""
        # ASSAY_TYPES is a _LazyDict
        value = ASSAY_TYPES["ChIP-seq"]
        assert isinstance(value, str)
        assert "ChIP-seq" in value
        assert "experiments" in value

    def test_lazy_dict_iter(self) -> None:
        """Lines 1085-1087: _LazyDict.__iter__ iterates over keys."""
        keys = list(iter(ASSAY_TYPES))
        assert isinstance(keys, list)
        assert len(keys) > 0
        assert "ChIP-seq" in keys

    def test_lazy_dict_keys(self) -> None:
        """Lines 1089-1091: _LazyDict.keys() returns keys."""
        keys = ASSAY_TYPES.keys()
        assert "ChIP-seq" in keys
        assert "RNA-seq" in keys

    def test_lazy_dict_values(self) -> None:
        """Lines 1093-1095: _LazyDict.values() returns values."""
        values = list(ASSAY_TYPES.values())
        assert isinstance(values, list)
        assert len(values) > 0
        # Values should be formatted strings
        assert all(isinstance(v, str) for v in values)

    def test_lazy_dict_items(self) -> None:
        """Test _LazyDict.items() returns key-value pairs."""
        items = list(ASSAY_TYPES.items())
        assert len(items) > 0
        for key, value in items[:5]:
            assert isinstance(key, str)
            assert isinstance(value, str)

    def test_lazy_dict_len(self) -> None:
        """Test _LazyDict.__len__ returns correct length."""
        length = len(ASSAY_TYPES)
        assert length > 0
        assert length == len(list(ASSAY_TYPES.keys()))

    def test_lazy_dict_contains(self) -> None:
        """Test _LazyDict.__contains__ checks membership."""
        assert "ChIP-seq" in ASSAY_TYPES
        assert "NONEXISTENT_ASSAY" not in ASSAY_TYPES

    def test_lazy_dict_get(self) -> None:
        """Test _LazyDict.get() with default."""
        value = ASSAY_TYPES.get("ChIP-seq")
        assert value is not None
        default_value = ASSAY_TYPES.get("NONEXISTENT", "default")
        assert default_value == "default"

    def test_lazy_list_getitem(self) -> None:
        """Lines 1122-1124: _LazyList.__getitem__ accesses items by index."""
        # TOP_BIOSAMPLES is a _LazyList
        first_item = TOP_BIOSAMPLES[0]
        assert isinstance(first_item, str)
        assert len(first_item) > 0

    def test_lazy_list_getitem_negative_index(self) -> None:
        """Test _LazyList.__getitem__ with negative index."""
        last_item = TOP_BIOSAMPLES[-1]
        assert isinstance(last_item, str)

    def test_lazy_list_iter(self) -> None:
        """Test _LazyList.__iter__ iterates over items."""
        items = list(iter(TOP_BIOSAMPLES))
        assert isinstance(items, list)
        assert len(items) > 0

    def test_lazy_list_len(self) -> None:
        """Test _LazyList.__len__ returns correct length."""
        length = len(TOP_BIOSAMPLES)
        assert length > 0
        assert length == 50  # Default limit for TOP_BIOSAMPLES

    def test_lazy_list_contains(self) -> None:
        """Test _LazyList.__contains__ checks membership."""
        # Get first biosample to check
        first = TOP_BIOSAMPLES[0]
        assert first in TOP_BIOSAMPLES
        assert "NONEXISTENT_BIOSAMPLE_XYZ" not in TOP_BIOSAMPLES

    def test_top_targets_lazy_list(self) -> None:
        """Test TOP_TARGETS lazy list access."""
        first_target = TOP_TARGETS[0]
        assert isinstance(first_target, str)
        length = len(TOP_TARGETS)
        assert length == 40  # Default limit for TOP_TARGETS

    def test_life_stages_lazy_list(self) -> None:
        """Test LIFE_STAGES lazy list access."""
        first_stage = LIFE_STAGES[0]
        assert isinstance(first_stage, str)
        assert "adult" in LIFE_STAGES

    def test_common_labs_lazy_list(self) -> None:
        """Test COMMON_LABS lazy list access."""
        first_lab = COMMON_LABS[0]
        assert isinstance(first_lab, str)
        length = len(COMMON_LABS)
        assert length == 20  # Default limit for COMMON_LABS


# =============================================================================
# Generic Slim Type Tests
# =============================================================================


class TestSlimTypes:
    """Tests for SLIM_TYPES configuration and generic slim functions."""

    def test_slim_types_config_exists(self) -> None:
        """Test that SLIM_TYPES configuration contains all four types."""
        assert "organ" in SLIM_TYPES
        assert "cell" in SLIM_TYPES
        assert "developmental" in SLIM_TYPES
        assert "system" in SLIM_TYPES

    def test_slim_types_has_required_keys(self) -> None:
        """Test that each slim type has required configuration keys."""
        for slim_type, config in SLIM_TYPES.items():
            assert "json_key" in config, f"{slim_type} missing json_key"
            assert "display_prefix" in config, f"{slim_type} missing display_prefix"
            assert "description" in config, f"{slim_type} missing description"

    def test_get_slim_categories_organ(self) -> None:
        """Test get_slim_categories for organ type."""
        organs = get_slim_categories("organ")
        assert isinstance(organs, list)
        assert len(organs) > 0
        name, count = organs[0]
        assert isinstance(name, str)
        assert isinstance(count, int)
        assert count > 0

    def test_get_slim_categories_cell(self) -> None:
        """Test get_slim_categories for cell type."""
        cells = get_slim_categories("cell")
        assert isinstance(cells, list)
        assert len(cells) > 0
        # Should include common cell types
        cell_names = [name for name, _ in cells]
        assert any("cell" in name.lower() for name in cell_names)

    def test_get_slim_categories_developmental(self) -> None:
        """Test get_slim_categories for developmental type."""
        layers = get_slim_categories("developmental")
        assert isinstance(layers, list)
        # Should have ~3 germ layers (mesoderm, ectoderm, endoderm)
        assert len(layers) >= 3
        layer_names = [name for name, _ in layers]
        expected = {"mesoderm", "ectoderm", "endoderm"}
        found = set(layer_names) & expected
        assert len(found) == 3, f"Expected all germ layers, found {layer_names}"

    def test_get_slim_categories_system(self) -> None:
        """Test get_slim_categories for system type."""
        systems = get_slim_categories("system")
        assert isinstance(systems, list)
        assert len(systems) > 0
        system_names = [name for name, _ in systems]
        # Should contain common body systems
        assert any("nervous" in name.lower() for name in system_names)
        assert any("immune" in name.lower() for name in system_names)

    def test_get_slim_categories_invalid_type(self) -> None:
        """Test that invalid slim type raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="Unknown slim type"):
            get_slim_categories("invalid_type")

    def test_get_slim_categories_ordered_by_count(self) -> None:
        """Test that slim categories are ordered by experiment count."""
        for slim_type in SLIM_TYPES.keys():
            categories = get_slim_categories(slim_type)
            counts = [count for name, count in categories]
            assert counts == sorted(
                counts, reverse=True
            ), f"{slim_type} categories not ordered by count"

    def test_get_slim_category_names(self) -> None:
        """Test get_slim_category_names returns list of names."""
        for slim_type in SLIM_TYPES.keys():
            names = get_slim_category_names(slim_type)
            assert isinstance(names, list)
            assert len(names) > 0
            assert all(isinstance(name, str) for name in names)

    def test_get_biosamples_for_slim(self) -> None:
        """Test get_biosamples_for_slim returns biosamples for a category."""
        # Test with organ
        brain_samples = get_biosamples_for_slim("organ", "brain")
        assert isinstance(brain_samples, list)
        assert len(brain_samples) > 0
        for name, count in brain_samples:
            assert isinstance(name, str)
            assert isinstance(count, int)

    def test_get_biosamples_for_slim_unknown_category(self) -> None:
        """Test get_biosamples_for_slim returns empty list for unknown category."""
        result = get_biosamples_for_slim("organ", "nonexistent_organ")
        assert result == []

    def test_build_biosample_to_slim(self) -> None:
        """Test build_biosample_to_slim builds reverse mapping."""
        for slim_type in SLIM_TYPES.keys():
            mapping = build_biosample_to_slim(slim_type)
            assert isinstance(mapping, dict)
            # Should have biosamples mapped
            if mapping:  # Some mappings may be empty in test data
                first_key = next(iter(mapping.keys()))
                assert isinstance(first_key, str)
                assert isinstance(mapping[first_key], list)

    def test_get_primary_slim_for_biosample(self) -> None:
        """Test get_primary_slim_for_biosample returns primary category."""
        # Test with a known biosample
        result = get_primary_slim_for_biosample("organ", "cerebellum")
        if result:  # May be None if not in mapping
            assert isinstance(result, str)

    def test_get_primary_slim_for_unknown_biosample(self) -> None:
        """Test get_primary_slim_for_biosample returns None for unknown."""
        result = get_primary_slim_for_biosample("organ", "nonexistent_sample")
        assert result is None

    def test_get_all_slims_for_biosample(self) -> None:
        """Test get_all_slims_for_biosample returns list of categories."""
        result = get_all_slims_for_biosample("organ", "cerebellum")
        assert isinstance(result, list)
        if result:
            assert all(isinstance(cat, str) for cat in result)

    def test_get_slim_display_name(self) -> None:
        """Test get_slim_display_name returns display names."""
        # Test organ
        assert get_slim_display_name("organ", "bodily fluid") == "Blood / Bodily Fluid"
        # Test cell
        assert "Cells" in get_slim_display_name("cell", "hematopoietic cell")
        # Test developmental
        assert "Mesoderm" in get_slim_display_name("developmental", "mesoderm")
        # Test system
        assert get_slim_display_name("system", "immune system") == "Immune System"


# =============================================================================
# Cell Slims Tests
# =============================================================================


class TestCellSlims:
    """Tests for cell_slims functions."""

    def test_get_cell_types_returns_list(self) -> None:
        """Test that get_cell_types returns ordered list of tuples."""
        cells = get_cell_types()
        assert isinstance(cells, list)
        assert len(cells) > 0
        name, count = cells[0]
        assert isinstance(name, str)
        assert isinstance(count, int)

    def test_get_cell_types_ordered_by_count(self) -> None:
        """Test that cell types are ordered by experiment count."""
        cells = get_cell_types()
        counts = [count for name, count in cells]
        assert counts == sorted(counts, reverse=True)

    def test_get_cell_types_contains_common_types(self) -> None:
        """Test that cell types contains common cell classifications."""
        cells = get_cell_types()
        cell_names = [name for name, _ in cells]
        # Should contain common cell types
        expected_patterns = ["cell", "stem", "epithelial", "hematopoietic"]
        for pattern in expected_patterns:
            assert any(
                pattern in name.lower() for name in cell_names
            ), f"Expected '{pattern}' in cell types"

    def test_get_cell_type_names(self) -> None:
        """Test get_cell_type_names returns list of names."""
        names = get_cell_type_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(name, str) for name in names)

    def test_get_biosamples_for_cell_type(self) -> None:
        """Test getting biosamples for a cell type."""
        cells = get_cell_types()
        if cells:
            first_cell = cells[0][0]
            biosamples = get_biosamples_for_cell_type(first_cell)
            assert isinstance(biosamples, list)

    def test_get_primary_cell_type_for_biosample(self) -> None:
        """Test getting primary cell type for biosample."""
        # K562 is a cancer cell line
        result = get_primary_cell_type_for_biosample("K562")
        if result:
            assert isinstance(result, str)

    def test_build_biosample_to_cell_types(self) -> None:
        """Test reverse mapping from biosample to cell types."""
        mapping = build_biosample_to_cell_types()
        assert isinstance(mapping, dict)

    def test_cell_type_display_names_mapping(self) -> None:
        """Test CELL_TYPE_DISPLAY_NAMES has valid entries."""
        assert isinstance(CELL_TYPE_DISPLAY_NAMES, dict)
        for key, value in CELL_TYPE_DISPLAY_NAMES.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    def test_get_cell_type_display_name(self) -> None:
        """Test get_cell_type_display_name returns display names."""
        result = get_cell_type_display_name("hematopoietic cell")
        assert result == "Blood/Immune Cells"
        # Unknown cell type should be title-cased
        result = get_cell_type_display_name("unknown_cell")
        assert result == "Unknown Cell"


# =============================================================================
# Developmental Slims Tests
# =============================================================================


class TestDevelopmentalSlims:
    """Tests for developmental_slims functions."""

    def test_get_developmental_layers_returns_list(self) -> None:
        """Test that get_developmental_layers returns list of tuples."""
        layers = get_developmental_layers()
        assert isinstance(layers, list)
        assert len(layers) >= 3  # At least 3 germ layers

    def test_developmental_layers_contains_germ_layers(self) -> None:
        """Test that all three germ layers are present."""
        layers = get_developmental_layers()
        layer_names = [name for name, count in layers]
        expected = {"mesoderm", "ectoderm", "endoderm"}
        assert expected.issubset(
            set(layer_names)
        ), f"Missing germ layers: {layer_names}"

    def test_get_developmental_layer_names(self) -> None:
        """Test get_developmental_layer_names returns list of names."""
        names = get_developmental_layer_names()
        assert isinstance(names, list)
        assert "mesoderm" in names
        assert "ectoderm" in names
        assert "endoderm" in names

    def test_get_biosamples_for_developmental_layer(self) -> None:
        """Test getting biosamples for a developmental layer."""
        biosamples = get_biosamples_for_developmental_layer("mesoderm")
        assert isinstance(biosamples, list)
        assert len(biosamples) > 0  # Mesoderm should have many biosamples

    def test_get_primary_developmental_layer_for_biosample(self) -> None:
        """Test getting primary developmental layer for biosample."""
        # Brain-derived tissue should be ectoderm
        result = get_primary_developmental_layer_for_biosample("cerebellum")
        if result:
            assert result == "ectoderm"

    def test_build_biosample_to_developmental_layers(self) -> None:
        """Test reverse mapping from biosample to developmental layers."""
        mapping = build_biosample_to_developmental_layers()
        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    def test_developmental_display_names_mapping(self) -> None:
        """Test DEVELOPMENTAL_DISPLAY_NAMES has valid entries."""
        assert isinstance(DEVELOPMENTAL_DISPLAY_NAMES, dict)
        assert "mesoderm" in DEVELOPMENTAL_DISPLAY_NAMES
        assert "ectoderm" in DEVELOPMENTAL_DISPLAY_NAMES
        assert "endoderm" in DEVELOPMENTAL_DISPLAY_NAMES

    def test_get_developmental_display_name(self) -> None:
        """Test get_developmental_display_name returns display names."""
        assert "Mesoderm" in get_developmental_display_name("mesoderm")
        assert "Ectoderm" in get_developmental_display_name("ectoderm")
        assert "Endoderm" in get_developmental_display_name("endoderm")


# =============================================================================
# System Slims Tests
# =============================================================================


class TestSystemSlims:
    """Tests for system_slims functions."""

    def test_get_body_systems_returns_list(self) -> None:
        """Test that get_body_systems returns list of tuples."""
        systems = get_body_systems()
        assert isinstance(systems, list)
        assert len(systems) > 0
        name, count = systems[0]
        assert isinstance(name, str)
        assert isinstance(count, int)

    def test_get_body_systems_ordered_by_count(self) -> None:
        """Test that body systems are ordered by experiment count."""
        systems = get_body_systems()
        counts = [count for name, count in systems]
        assert counts == sorted(counts, reverse=True)

    def test_get_body_systems_contains_known_systems(self) -> None:
        """Test that known body systems are present."""
        systems = get_body_systems()
        system_names = [name.lower() for name, count in systems]
        # Should contain common body systems
        expected_keywords = ["nervous", "immune", "digestive", "respiratory"]
        for keyword in expected_keywords:
            assert any(
                keyword in s for s in system_names
            ), f"Expected '{keyword}' in system names"

    def test_get_body_system_names(self) -> None:
        """Test get_body_system_names returns list of names."""
        names = get_body_system_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(name, str) for name in names)

    def test_get_biosamples_for_body_system(self) -> None:
        """Test getting biosamples for a body system."""
        systems = get_body_systems()
        if systems:
            first_system = systems[0][0]
            biosamples = get_biosamples_for_body_system(first_system)
            assert isinstance(biosamples, list)

    def test_get_primary_body_system_for_biosample(self) -> None:
        """Test getting primary body system for biosample."""
        result = get_primary_body_system_for_biosample("heart")
        if result:
            assert isinstance(result, str)

    def test_build_biosample_to_body_systems(self) -> None:
        """Test reverse mapping from biosample to body systems."""
        mapping = build_biosample_to_body_systems()
        assert isinstance(mapping, dict)

    def test_system_display_names_mapping(self) -> None:
        """Test SYSTEM_DISPLAY_NAMES has valid entries."""
        assert isinstance(SYSTEM_DISPLAY_NAMES, dict)
        assert "immune system" in SYSTEM_DISPLAY_NAMES
        assert "nervous" in "".join(SYSTEM_DISPLAY_NAMES.keys()).lower()

    def test_get_system_display_name(self) -> None:
        """Test get_system_display_name returns display names."""
        assert get_system_display_name("immune system") == "Immune System"
        # Unknown system should be title-cased
        result = get_system_display_name("unknown_system")
        assert result == "Unknown System"
