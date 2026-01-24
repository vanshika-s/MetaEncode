# tests/test_api/test_encode_client.py
"""Tests for the ENCODE API client."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.api.encode_client import EncodeClient, RateLimiter


class TestRateLimiter:
    """Test suite for RateLimiter."""

    def test_rate_limiter_initialization(self):
        """Test that rate limiter initializes correctly."""
        limiter = RateLimiter(max_requests=10, window_seconds=1.0)
        assert limiter.max_requests == 10
        assert limiter.window == 1.0

    def test_rate_limiter_allows_requests_under_limit(self):
        """Test that rate limiter allows requests under the limit."""
        limiter = RateLimiter(max_requests=5, window_seconds=1.0)
        # Should not block for fewer than max_requests
        for _ in range(4):
            limiter.wait_if_needed()
        # If we get here without blocking too long, test passes


class TestEncodeClient:
    """Test suite for EncodeClient."""

    def test_client_initialization(self):
        """Test that client initializes correctly."""
        client = EncodeClient()
        assert client.BASE_URL == "https://www.encodeproject.org"
        assert client.RATE_LIMIT == 10
        assert client._rate_limiter is not None

    def test_build_search_url(self):
        """Test URL building."""
        client = EncodeClient()
        url = client._build_search_url({"type": "Experiment", "limit": 10})
        assert "https://www.encodeproject.org/search/?" in url
        assert "type=Experiment" in url
        assert "limit=10" in url

    def test_parse_experiment(self, sample_experiment_data):
        """Test experiment parsing."""
        client = EncodeClient()
        parsed = client._parse_experiment(sample_experiment_data)
        assert parsed["accession"] == "ENCSR000AAA"
        assert parsed["assay_term_name"] == "ChIP-seq"
        assert parsed["replicate_count"] == 2
        assert parsed["file_count"] == 2

    @patch("requests.Session.get")
    def test_fetch_experiments_returns_dataframe(
        self, mock_get, sample_experiment_data
    ):
        """Test that fetch_experiments returns a DataFrame."""
        mock_response = Mock()
        mock_response.json.return_value = {"@graph": [sample_experiment_data]}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = EncodeClient()
        df = client.fetch_experiments(limit=5)

        assert isinstance(df, pd.DataFrame)
        assert "accession" in df.columns
        assert len(df) == 1
        assert df.iloc[0]["accession"] == "ENCSR000AAA"

    @patch("requests.Session.get")
    def test_fetch_experiment_by_accession(self, mock_get, sample_experiment_data):
        """Test fetching single experiment by accession."""
        mock_response = Mock()
        mock_response.json.return_value = sample_experiment_data
        mock_response.raise_for_status = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = EncodeClient()
        result = client.fetch_experiment_by_accession("ENCSR000AAA")

        assert "accession" in result
        assert result["accession"] == "ENCSR000AAA"

    @patch("requests.Session.get")
    def test_search_returns_results(self, mock_get, sample_experiment_data):
        """Test that search returns matching results."""
        mock_response = Mock()
        mock_response.json.return_value = {"@graph": [sample_experiment_data]}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = EncodeClient()
        results = client.search("K562", limit=5)

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 1

    @patch("requests.Session.get")
    def test_fetch_experiment_not_found(self, mock_get):
        """Test that ValueError is raised for non-existent accession."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        client = EncodeClient()
        with pytest.raises(ValueError, match="not found"):
            client.fetch_experiment_by_accession("NONEXISTENT")


# ============================================================================
# RateLimiter Sleep Tests (Lines 50-52)
# ============================================================================


class TestRateLimiterSleep:
    """Tests for RateLimiter sleep branch."""

    def test_rate_limiter_triggers_sleep_at_limit(self):
        """Lines 50-52: Sleep when rate limit exceeded."""
        with patch("src.api.encode_client.time.sleep") as mock_sleep:
            limiter = RateLimiter(max_requests=2, window_seconds=1.0)
            limiter.wait_if_needed()
            limiter.wait_if_needed()
            limiter.wait_if_needed()  # Should trigger sleep
            mock_sleep.assert_called()

    def test_rate_limiter_calculates_sleep_time(self):
        """Test that sleep time is calculated correctly."""
        with patch("src.api.encode_client.time.sleep") as mock_sleep:
            limiter = RateLimiter(max_requests=1, window_seconds=1.0)
            limiter.wait_if_needed()
            limiter.wait_if_needed()  # Should trigger sleep
            # Sleep time should be positive (roughly 1 second minus elapsed time)
            assert mock_sleep.called
            call_args = mock_sleep.call_args[0][0]
            assert 0 < call_args <= 1.0


# ============================================================================
# fetch_experiments Filter Tests (Lines 110-120)
# ============================================================================


class TestFetchExperimentsFilters:
    """Tests for fetch_experiments filter parameters."""

    @patch("requests.Session.get")
    def test_fetch_with_assay_type_filter(self, mock_get):
        """Line 110: assay_type parameter adds assay_term_name to URL."""
        mock_response = Mock()
        mock_response.json.return_value = {"@graph": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = EncodeClient()
        client.fetch_experiments(assay_type="ChIP-seq", limit=10)

        # Verify URL contains the assay_term_name parameter
        call_url = mock_get.call_args[0][0]
        assert "assay_term_name=ChIP-seq" in call_url

    @patch("requests.Session.get")
    def test_fetch_with_organism_filter(self, mock_get):
        """Lines 112-114: organism parameter adds nested path to URL."""
        mock_response = Mock()
        mock_response.json.return_value = {"@graph": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = EncodeClient()
        client.fetch_experiments(organism="Homo sapiens", limit=10)

        # Verify URL contains the nested organism path
        call_url = mock_get.call_args[0][0]
        assert "replicates.library.biosample.donor.organism.scientific_name" in call_url
        assert "Homo+sapiens" in call_url or "Homo%20sapiens" in call_url

    @patch("requests.Session.get")
    def test_fetch_with_biosample_filter(self, mock_get):
        """Line 116: biosample parameter adds biosample_ontology.term_name to URL."""
        mock_response = Mock()
        mock_response.json.return_value = {"@graph": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = EncodeClient()
        client.fetch_experiments(biosample="K562", limit=10)

        # Verify URL contains the biosample_ontology.term_name parameter
        call_url = mock_get.call_args[0][0]
        assert "biosample_ontology.term_name=K562" in call_url

    @patch("requests.Session.get")
    def test_fetch_with_limit_zero_uses_all(self, mock_get):
        """Lines 118-120: limit=0 sets limit to 'all'."""
        mock_response = Mock()
        mock_response.json.return_value = {"@graph": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = EncodeClient()
        client.fetch_experiments(limit=0)

        # Verify URL contains limit=all
        call_url = mock_get.call_args[0][0]
        assert "limit=all" in call_url

    @patch("requests.Session.get")
    def test_fetch_with_target_filter(self, mock_get):
        """Line 124: target parameter adds target.label to URL."""
        mock_response = Mock()
        mock_response.json.return_value = {"@graph": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = EncodeClient()
        client.fetch_experiments(target="H3K27ac", limit=10)

        # Verify URL contains the target.label parameter
        call_url = mock_get.call_args[0][0]
        assert "target.label=H3K27ac" in call_url

    @patch("requests.Session.get")
    def test_fetch_with_life_stage_filter(self, mock_get):
        """Line 126: life_stage parameter adds nested path to URL."""
        mock_response = Mock()
        mock_response.json.return_value = {"@graph": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = EncodeClient()
        client.fetch_experiments(life_stage="adult", limit=10)

        # Verify URL contains the life_stage nested path
        call_url = mock_get.call_args[0][0]
        assert "replicates.library.biosample.life_stage=adult" in call_url

    @patch("requests.Session.get")
    def test_fetch_with_search_term_filter(self, mock_get):
        """Line 128: search_term parameter adds searchTerm to URL."""
        mock_response = Mock()
        mock_response.json.return_value = {"@graph": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = EncodeClient()
        client.fetch_experiments(search_term="enhancer", limit=10)

        # Verify URL contains the searchTerm parameter
        call_url = mock_get.call_args[0][0]
        assert "searchTerm=enhancer" in call_url


# ============================================================================
# _parse_experiment Edge Cases (Lines 228-280)
# ============================================================================


class TestParseExperimentEdgeCases:
    """Tests for _parse_experiment edge cases."""

    def test_parse_lab_as_dict(self):
        """Lines 227-228: Lab as dictionary with title."""
        client = EncodeClient()
        data = {"lab": {"title": "Snyder Lab", "name": "snyder"}}
        result = client._parse_experiment(data)
        assert result["lab"] == "Snyder Lab"

    def test_parse_lab_as_dict_fallback_to_name(self):
        """Lines 227-228: Lab as dictionary without title, fallback to name."""
        client = EncodeClient()
        data = {"lab": {"name": "snyder"}}
        result = client._parse_experiment(data)
        assert result["lab"] == "snyder"

    def test_parse_lab_as_path_string(self):
        """Lines 229-232: Lab as path string (contains '/')."""
        client = EncodeClient()
        data = {"lab": "/labs/encode-consortium/"}
        result = client._parse_experiment(data)
        assert result["lab"] == "encode-consortium"

    def test_parse_lab_as_plain_string(self):
        """Lines 233-234: Lab as plain string (no slash)."""
        client = EncodeClient()
        data = {"lab": "Snyder Lab"}
        result = client._parse_experiment(data)
        assert result["lab"] == "Snyder Lab"

    def test_parse_biosample_ontology_not_dict(self):
        """Lines 240-241: biosample_ontology as string (not dict)."""
        client = EncodeClient()
        data = {"biosample_ontology": "/biosample-types/cell_line/"}
        result = client._parse_experiment(data)
        assert result["biosample_term_name"] == ""

    def test_parse_organism_from_replicates_nested(self):
        """Lines 244-260: Organism from deeply nested replicates structure."""
        client = EncodeClient()
        data = {
            "replicates": [
                {
                    "library": {
                        "biosample": {
                            "donor": {
                                "organism": {
                                    "name": "human",
                                    "scientific_name": "Homo sapiens",
                                }
                            }
                        }
                    }
                }
            ]
        }
        result = client._parse_experiment(data)
        assert result["organism"] == "human"

    def test_parse_organism_from_biosample_ontology_dict(self):
        """Lines 263-268: Organism from biosample_ontology.organism dict."""
        client = EncodeClient()
        data = {
            "biosample_ontology": {
                "organism": {"name": "mouse", "scientific_name": "Mus musculus"}
            }
        }
        result = client._parse_experiment(data)
        assert result["organism"] == "mouse"

    def test_parse_organism_string_in_biosample_ontology(self):
        """Lines 269-270: Organism as string in biosample_ontology."""
        client = EncodeClient()
        data = {"biosample_ontology": {"organism": "mouse"}}
        result = client._parse_experiment(data)
        assert result["organism"] == "mouse"

    def test_parse_organism_top_level_dict(self):
        """Lines 274-278: Top-level organism as dictionary."""
        client = EncodeClient()
        data = {"organism": {"name": "human", "scientific_name": "Homo sapiens"}}
        result = client._parse_experiment(data)
        assert result["organism"] == "human"

    def test_parse_organism_top_level_string(self):
        """Lines 279-280: Top-level organism as string."""
        client = EncodeClient()
        data = {"organism": "human"}
        result = client._parse_experiment(data)
        assert result["organism"] == "human"
