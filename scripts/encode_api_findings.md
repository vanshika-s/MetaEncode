# ENCODE API Findings

This document summarizes findings from exploring the ENCODE API and schema to understand what options actually exist.

## Assay Types (97 total from dev branch)

**Official ENCODE assay_term_name values** (from `mixins.json`):

```
3' RACE, 4C, 5' RACE, 5' RLM RACE, 5C, ATAC-seq, Bru-seq, BruChase-seq,
BruUV-seq, CAGE, capture Hi-C, ChIA-PET, ChIP-seq, Circulome-seq, Clone-seq,
comparative genomic hybridization by array, CRISPR genome editing followed by RNA-seq,
CRISPRi followed by RNA-seq, CUT&RUN, CUT&Tag, direct RNA-seq,
DNA methylation profiling by array assay, DNA-PET, DNase-seq, eCLIP, FAIRE-seq,
genetic modification followed by DNase-seq, genomic perturbation followed by RT-qPCR,
genotype phasing by HiC, GRO-cap, GRO-seq, HiC, iCLIP, icLASER, icSHAPE,
LC/MS label-free quantitative proteomics, LC-MS/MS isobaric label quantitative proteomics,
long read RNA-seq, long read single-cell RNA-seq, MeDIP-seq, microRNA counts,
microRNA-seq, Mint-ChIP-seq, MNase-seq, MRE-seq, PAS-seq, PLAC-seq,
polyA plus RNA-seq, polyA minus RNA-seq, PRO-cap, PRO-seq,
protein sequencing by tandem mass spectrometry assay, RAMPAGE, Repli-chip,
Repli-seq, Ribo-seq, RIP-chip, RIP-seq, RNA Bind-n-Seq, RNA-PET, RNA-seq,
RRBS, seqFISH, shRNA knockdown followed by RNA-seq, single-cell RNA sequencing assay,
single-nucleus ATAC-seq, siRNA knockdown followed by RNA-seq, small RNA-seq,
SPRITE, SPRITE-IP, Switchgear, TAB-seq, transcription profiling by array assay,
whole genome sequencing assay, whole-genome shotgun bisulfite sequencing
```

### Key Notes:
- **"HiC"** not "Hi-C" - ENCODE uses "HiC" as the canonical spelling
- **"capture Hi-C"** exists (with space and hyphen)
- **"4C" and "5C"** are valid
- **"CUT&RUN" and "CUT&Tag"** are valid (in dev branch)

## Organisms (4 main)

| Key | Scientific Name | Assembly |
|-----|-----------------|----------|
| human | Homo sapiens | hg38 (GRCh38) |
| mouse | Mus musculus | mm10 (GRCm38), mm39 newer |
| fly | Drosophila melanogaster | dm6 |
| worm | Caenorhabditis elegans | ce11 |

## Cell Types (Tier Classification)

### Tier 1 (Highest Priority)
- GM12878 - B-lymphocyte, lymphoblastoid
- K562 - Chronic myelogenous leukemia
- H1-hESC - Human embryonic stem cells

### Tier 2
- A549 - Lung carcinoma
- HeLa-S3 - Cervical carcinoma
- HepG2 - Hepatocellular carcinoma
- HUVEC - Umbilical vein endothelial
- IMR-90 - Fetal lung fibroblasts
- MCF-7 - Breast adenocarcinoma
- SK-N-SH - Neuroblastoma

### Tier 3
100+ additional cell types including primary tissues, disease samples, etc.

## Developmental Stages / Age Handling

**IMPORTANT**: Developmental stages are NOT direct search parameters. They are stored in these biosample fields:

### Mouse-specific fields:
- **life_stage**: "embryonic", "postnatal", "adult", "unknown"
- **model_organism_age**: Free text like "P60", "E14.5", "8 weeks"
- **model_organism_age_units**: "day", "week", "month", "year", "stage"

### Fly-specific life_stage values:
embryonic, larva, first/second/third instar larva, wandering third instar larva, prepupa, pupa, adult

### Worm-specific life_stage values:
early embryonic, midembryonic, late embryonic, L1-L4 larva, dauer, adult, mixed stage

### How to Search by Developmental Stage:
1. Use `replicates.library.biosample.life_stage=adult` for life stage filtering
2. Search in descriptions using free-text search
3. Use the Mouse Development Matrix endpoint for organized time course data

## Search Approaches

### 1. Structured Search (fetch_experiments)
Use API parameters for exact matching:
- `assay_term_name=ChIP-seq`
- `replicates.library.biosample.donor.organism.scientific_name=Mus+musculus`
- `biosample_ontology.term_name=K562`
- `replicates.library.biosample.life_stage=adult`

### 2. Free-text Search (searchTerm)
Uses ElasticSearch-style matching:
- Good for finding keywords in descriptions
- NOT good for structured queries
- Can cause 404 errors with unsupported terms

### 3. Post-filtering
Fetch broader results then filter locally on:
- Description text (for age like "P60", "8 week", "E14.5")
- Other metadata fields

## Recommendations for vocabularies.py

1. **Assay Types**: Update to use exact ENCODE values (97 types)
2. **Organisms**: Current 4 organisms are correct
3. **Cell Types**: Order by Tier (1, 2, 3) for common options
4. **Developmental Stages**:
   - Remove as search parameter
   - Use for post-filtering on description text
   - Or use life_stage API parameter ("embryonic", "postnatal", "adult")

## Search Implementation Recommendations

1. **Change from `client.search()` to `client.fetch_experiments()`** for primary filters
2. **Use proper API parameters**:
   - `assay_term_name` instead of searchTerm
   - `biosample_ontology.term_name` for tissue/cell type
   - `replicates.library.biosample.life_stage` for broad life stage
3. **Post-filter for developmental stages**:
   - Search description for "P60", "E14.5", "8 week", etc.
   - Use ML similarity for related matches
4. **Remove hallucinated options**:
   - Only include assay types that exist in ENCODE schema
   - Order options by frequency of usage in ENCODE

## Sources

- ENCODE REST API: https://www.encodeproject.org/help/rest-api/
- ENCODE Schema GitHub (dev): https://github.com/ENCODE-DCC/encoded/tree/dev/src/encoded/schemas
- ENCODE Data Organization: https://www.encodeproject.org/help/data-organization/
- ENCODE Pipelines: https://www.encodeproject.org/pipelines/
- ENCODE Mouse Development Matrix: https://www.encodeproject.org/mouse-development-matrix/
