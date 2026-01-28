# src/api/encode_client.py
"""ENCODE REST API client for fetching experiment metadata.

This module provides a client for interacting with the ENCODE Data Coordination
Center (DCC) REST API. It handles rate limiting, pagination, and conversion of
JSON responses to pandas DataFrames.

Reference: https://www.encodeproject.org/help/rest-api/
"""

import time
from typing import Any, Optional
from urllib.parse import urlencode

import pandas as pd
import requests


class RateLimiter:
    """Enforce ENCODE's 10 requests/second limit.

    Tracks request timestamps and sleeps when necessary to avoid
    exceeding the rate limit.

    Example:
        >>> limiter = RateLimiter(max_requests=10, window_seconds=1)
        >>> limiter.wait_if_needed()  # Call before each request
    """

    def __init__(self, max_requests: int = 10, window_seconds: float = 1.0) -> None:
        """Initialize the rate limiter.

        Args:
            max_requests: Maximum requests allowed per window.
            window_seconds: Time window in seconds.
        """
        self.max_requests = max_requests
        self.window = window_seconds
        self._request_times: list[float] = []

    def wait_if_needed(self) -> None:
        """Wait if necessary to stay within rate limits."""
        now = time.time()

        # Remove requests outside the current window
        self._request_times = [t for t in self._request_times if now - t < self.window]

        if len(self._request_times) >= self.max_requests:
            # Need to wait until oldest request exits the window
            sleep_time = self.window - (now - self._request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._request_times.append(time.time())


class EncodeClient:
    """Client for interacting with the ENCODE REST API.

    Attributes:
        BASE_URL: Base URL for the ENCODE API.
        RATE_LIMIT: Maximum requests per second (API limit is 10).

    Example:
        >>> client = EncodeClient()
        >>> experiments = client.fetch_experiments(assay_type="ChIP-seq", limit=100)
        >>> print(experiments.head())
    """

    BASE_URL = "https://www.encodeproject.org"
    RATE_LIMIT = 10  # requests per second
    HEADERS = {"accept": "application/json"}

    def __init__(self) -> None:
        """Initialize the ENCODE API client."""
        self._session = requests.Session()
        self._session.headers.update(self.HEADERS)
        self._rate_limiter = RateLimiter(max_requests=self.RATE_LIMIT)

    def fetch_experiments(
        self,
        assay_type: Optional[str] = None,
        organism: Optional[str] = None,
        biosample: Optional[str] = None,
        target: Optional[str] = None,
        life_stage: Optional[str] = None,
        search_term: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch experiment metadata from ENCODE API.

        Args:
            assay_type: Filter by assay type (e.g., "ChIP-seq", "RNA-seq", "HiC").
            organism: Filter by organism (e.g., "Homo sapiens", "Mus musculus").
            biosample: Filter by biosample term name (e.g., "K562", "cerebellum").
            target: Filter by ChIP-seq target (e.g., "H3K27ac", "CTCF").
            life_stage: Filter by life stage (e.g., "adult", "embryonic").
            search_term: Free-text search term (e.g., "mouse cerebellum").
            limit: Maximum number of experiments to fetch (default 100, use 0 for all).

        Returns:
            DataFrame containing experiment metadata with columns:
            accession, description, assay_term_name, biosample_term_name,
            organism, lab, status, replicate_count, file_count.

        Raises:
            requests.RequestException: If the API request fails.
        """
        params: dict[str, Any] = {
            "type": "Experiment",
            "frame": "embedded",
            "format": "json",
        }

        if assay_type:
            params["assay_term_name"] = assay_type
        if organism:
            params["replicates.library.biosample.donor.organism.scientific_name"] = (
                organism
            )
        if biosample:
            params["biosample_ontology.term_name"] = biosample
        if target:
            params["target.label"] = target
        if life_stage:
            params["replicates.library.biosample.life_stage"] = life_stage
        if search_term:
            params["searchTerm"] = search_term
        if limit > 0:
            params["limit"] = limit
        else:
            params["limit"] = "all"

        url = self._build_search_url(params)
        self._rate_limiter.wait_if_needed()

        response = self._session.get(url, timeout=60)
        response.raise_for_status()

        data = response.json()
        experiments = data.get("@graph", [])

        # Parse each experiment into standardized format
        records = [self._parse_experiment(exp) for exp in experiments]

        return pd.DataFrame(records)

    def fetch_experiment_by_accession(self, accession: str) -> dict:
        """Fetch a single experiment by its accession number.

        Args:
            accession: ENCODE accession number (e.g., "ENCSR000AKS").

        Returns:
            Dictionary containing the parsed experiment metadata.

        Raises:
            requests.RequestException: If the API request fails.
            ValueError: If the accession is not found.
        """
        url = f"{self.BASE_URL}/experiments/{accession}/?frame=embedded&format=json"

        self._rate_limiter.wait_if_needed()

        response = self._session.get(url, timeout=30)

        if response.status_code == 404:
            raise ValueError(f"Experiment not found: {accession}")

        response.raise_for_status()

        data = response.json()
        return self._parse_experiment(data)

    def search(
        self,
        search_term: str,
        object_type: str = "Experiment",
        limit: int = 25,
    ) -> pd.DataFrame:
        """Search ENCODE for datasets matching a search term.

        Args:
            search_term: Text to search for.
            object_type: Type of object to search (default "Experiment").
            limit: Maximum results to return.

        Returns:
            DataFrame containing search results.
        """
        params: dict[str, Any] = {
            "searchTerm": search_term,
            "type": object_type,
            "frame": "embedded",
            "format": "json",
            "limit": limit,
        }

        url = self._build_search_url(params)
        self._rate_limiter.wait_if_needed()

        response = self._session.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        results = data.get("@graph", [])

        # Parse each result into standardized format
        records = [self._parse_experiment(exp) for exp in results]

        return pd.DataFrame(records)

    def _build_search_url(self, params: dict) -> str:
        """Build a search URL with the given parameters.

        Args:
            params: Dictionary of query parameters.

        Returns:
            Fully formed search URL.
        """
        query_string = urlencode(params)
        return f"{self.BASE_URL}/search/?{query_string}"

    def _parse_experiment(self, data: dict) -> dict:
        """Parse raw experiment JSON into standardized format.

        Extracts key fields from the nested ENCODE JSON structure,
        handling missing fields gracefully. Works with frame=embedded responses.

        Args:
            data: Raw JSON data from API response.

        Returns:
            Dictionary with standardized field names.
        """
        # Extract lab name from various formats
        lab_raw = data.get("lab", "")
        if isinstance(lab_raw, dict):
            lab = lab_raw.get("title", lab_raw.get("name", ""))
        elif isinstance(lab_raw, str) and "/" in lab_raw:
            # Format: "/labs/lab-name/"
            parts = lab_raw.strip("/").split("/")
            lab = parts[-1] if parts else ""
        else:
            lab = str(lab_raw) if lab_raw else ""

        # Extract biosample term name from nested structure
        biosample_ontology = data.get("biosample_ontology", {})
        if isinstance(biosample_ontology, dict):
            biosample_term_name = biosample_ontology.get("term_name", "")
        else:
            biosample_term_name = ""

        # Extract organism from replicates structure (most reliable with frame=embedded)
        # Path: replicates[0].library.biosample.donor.organism.name
        organism = ""
        replicates = data.get("replicates", [])
        if replicates and isinstance(replicates, list) and len(replicates) > 0:
            rep = replicates[0]
            if isinstance(rep, dict):
                library = rep.get("library", {})
                if isinstance(library, dict):
                    biosample = library.get("biosample", {})
                    if isinstance(biosample, dict):
                        donor = biosample.get("donor", {})
                        if isinstance(donor, dict):
                            org_data = donor.get("organism", {})
                            if isinstance(org_data, dict):
                                organism = str(
                                    org_data.get(
                                        "name", org_data.get("scientific_name", "")
                                    )
                                    or ""
                                )

        # Fallback: check biosample_ontology.organism
        if not organism and isinstance(biosample_ontology, dict):
            organism_data = biosample_ontology.get("organism", {})
            if isinstance(organism_data, dict):
                organism = str(
                    organism_data.get("name", organism_data.get("scientific_name", ""))
                    or ""
                )
            elif isinstance(organism_data, str):
                organism = organism_data

        # Fallback: check top-level organism field
        if not organism:
            organism_top = data.get("organism", {})
            if isinstance(organism_top, dict):
                organism = str(
                    organism_top.get("name", organism_top.get("scientific_name", ""))
                    or ""
                )
            elif isinstance(organism_top, str):
                organism = organism_top

        # Extract life_stage from replicates structure
        # Path: replicates[0].library.biosample.life_stage
        life_stage = ""
        if replicates and isinstance(replicates, list) and len(replicates) > 0:
            rep = replicates[0]
            if isinstance(rep, dict):
                library = rep.get("library", {})
                if isinstance(library, dict):
                    biosample = library.get("biosample", {})
                    if isinstance(biosample, dict):
                        life_stage = str(biosample.get("life_stage", "") or "")

        return {
            "accession": data.get("accession", ""),
            "description": data.get("description", ""),
            "title": data.get("title", ""),
            "assay_term_name": data.get("assay_term_name", ""),
            "biosample_term_name": biosample_term_name,
            "organism": organism,
            "life_stage": life_stage,
            "lab": lab,
            "status": data.get("status", ""),
            "replicate_count": len(replicates),
            "file_count": len(data.get("files", [])),
        }
