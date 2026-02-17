# MetaENCODE

**Discover related ENCODE datasets through metadata-driven similarity scoring.**

MetaENCODE is a Streamlit web application that helps researchers find related biological datasets from the [Encyclopedia of DNA Elements (ENCODE)](https://www.encodeproject.org/). It transforms dataset metadata into numeric representations using ML-based embeddings and similarity computation to rank and recommend datasets, reducing manual filtering and enabling exploratory science.

**DS3 x UBIC Collaborative Project** | Led by Vanshika + Isha

---

## Features

- **Dataset Search & Selection** — Search ENCODE experiments by assay type, organism, biosample, target, life stage, and free-text description with spell correction
- **Similarity Recommendations** — Select a seed dataset and get top-N similar experiments ranked by combined similarity score
- **Hierarchical Filtering** — Browse biosamples through organ system, cell type, germ layer, or body system classifications
- **Interactive Visualization** — Explore dataset relationships via UMAP, PCA, or t-SNE scatter plots colored by metadata attributes
- **ENCODE Portal Links** — Click through to the original ENCODE experiment pages

## Architecture

```
ENCODE REST API
    |
EncodeClient (fetch + rate limiting)
    |
MetadataProcessor (text cleaning, missing value imputation, ontology mapping)
    |
EmbeddingGenerator (SBERT: text -> 384-dim vectors)
    |
FeatureCombiner (weighted concatenation: text + categorical + numeric -> ~437-dim)
    |
CacheManager (persist precomputed data as pickle files)
    |
SimilarityEngine (cosine similarity via NearestNeighbors index)
    |
Streamlit UI (3 tabs: Search & Select, Similar Datasets, Visualize)
```

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | [Streamlit](https://streamlit.io/) |
| Text Embeddings | [Sentence Transformers](https://www.sbert.net/) (all-MiniLM-L6-v2) |
| Similarity | scikit-learn (cosine similarity, NearestNeighbors) |
| Visualization | Plotly + UMAP / PCA / t-SNE |
| Data Processing | pandas |
| API | requests (ENCODE REST API) |

## Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Zaki-1052/MetaENCODE.git
cd MetaENCODE

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Precompute Embeddings

Before running the app, precompute the dataset embeddings:

```bash
# Quick test (100 experiments)
python scripts/precompute_embeddings.py --limit 100

# Medium dataset (1000 experiments)
python scripts/precompute_embeddings.py --limit 1000

# Full dataset (all experiments, requires more time/memory)
python scripts/precompute_embeddings.py --limit all --batch-size 64

# Force refresh
python scripts/precompute_embeddings.py --limit 1000 --refresh
```

This fetches experiments from the ENCODE API, generates SBERT embeddings, combines features, and caches everything in `data/cache/`.

### Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Usage

1. **Search**: Use the sidebar filters (assay type, organism, biosample, etc.) to search ENCODE for experiments
2. **Select**: Click on a row in the search results to select a seed dataset
3. **Find Similar**: Switch to the "Similar Datasets" tab and click "Find Similar Datasets"
4. **Visualize**: Switch to the "Visualize" tab to see an interactive 2D embedding plot of datasets
5. **Explore**: Hover over points in the plot to see metadata; click accession links to visit ENCODE

## Project Structure

```
MetaENCODE/
├── app.py                         # Streamlit entry point
├── requirements.txt               # Python dependencies
├── src/
│   ├── api/
│   │   └── encode_client.py       # ENCODE REST API client with rate limiting
│   ├── ml/
│   │   ├── embeddings.py          # SBERT text embedding generation
│   │   ├── similarity.py          # Cosine similarity & nearest neighbor search
│   │   └── feature_combiner.py    # Weighted feature concatenation
│   ├── processing/
│   │   ├── metadata.py            # Text cleaning & metadata normalization
│   │   └── encoders.py            # Categorical (one-hot) & numeric (minmax) encoding
│   ├── ui/
│   │   ├── sidebar.py             # Sidebar filter controls
│   │   ├── handlers.py            # Search execution logic
│   │   ├── search_filters.py      # Filter state management & fuzzy matching
│   │   ├── vocabularies.py        # ENCODE vocabulary definitions
│   │   ├── autocomplete.py        # Autocomplete logic
│   │   ├── formatters.py          # Display formatting utilities
│   │   ├── components/            # Session state & cached initializers
│   │   └── tabs/                  # Search, Similar, Visualize tab UIs
│   ├── utils/
│   │   ├── cache.py               # File-based caching with atomic writes
│   │   └── spell_check.py         # Biology-aware spell correction
│   └── visualization/
│       └── plots.py               # UMAP/PCA/t-SNE + Plotly scatter plots
├── tests/                         # Comprehensive test suite (649 tests)
├── scripts/                       # Precomputation & utility scripts
├── data/
│   ├── encode_facets_raw.json     # Vocabulary source (27,398 experiments)
│   └── cache/                     # Precomputed embeddings & metadata
└── docs/
    └── PRD_COMPLIANCE.md          # PRD compliance report
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run specific module tests
python -m pytest tests/test_ml/ -v
python -m pytest tests/test_api/ -v
```

Note: Some tests require `sentence-transformers` (PyTorch) and `umap-learn` to be installed. Without these heavy ML dependencies, 619 out of 649 tests pass.

## Similarity Scoring

MetaENCODE combines multiple feature types into a single vector per dataset:

| Feature | Weight | Method |
|---|---|---|
| Text (description + title) | 0.50 | SBERT embeddings (384-dim) |
| Assay type | 0.20 | One-hot encoding |
| Organism | 0.15 | One-hot encoding |
| Cell type / Biosample | 0.10 | One-hot encoding |
| Lab | 0.03 | One-hot encoding |
| Numeric (replicates, files) | 0.02 | Min-max normalization |

Weights are applied via `sqrt(weight)` scaling so that cosine similarity contributions are proportional to the configured weights.

## ENCODE API

MetaENCODE uses the [ENCODE REST API](https://www.encodeproject.org/help/rest-api/) for data access:

- **Base URL:** `https://www.encodeproject.org/`
- **Rate Limit:** 10 requests/second (enforced by `RateLimiter`)
- **Authentication:** None required for public data
- **Key Parameters:** `type=Experiment`, `frame=embedded`, `format=json`

## Known Limitations

- The precomputation step requires network access to the ENCODE API
- Full dataset precomputation requires significant memory (~40GB for all experiments on HPC)
- UMAP/t-SNE visualizations can be slow for large datasets; PCA is faster
- Spell correction requires `symspellpy` and `jellyfish` packages

## License

This project is part of the DS3 x UBIC collaborative program at UCSD.