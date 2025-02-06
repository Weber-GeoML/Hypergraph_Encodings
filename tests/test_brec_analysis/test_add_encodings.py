"""Tests for add_encodings.py."""

from typing import Any

import numpy as np
import pytest

from brec_analysis.add_encodings import get_encodings
from encodings_hnns.encodings import HypergraphEncodings


@pytest.fixture
def sample_hypergraph() -> dict[str, dict[str, Any] | int | np.ndarray]:
    """Create a sample hypergraph for testing."""
    hg: dict[str, dict[str, Any] | int | np.ndarray] = {
        "hypergraph": {
            "yellow": [0, 1, 2],
            "red": [1, 2],
            "green": [2, 4, 5],
            "blue": [3, 4],
        },
        "features": np.zeros((6, 1)),  # For nodes 0-5
        "labels": {},
        "n": 6,
    }
    return hg


@pytest.fixture
def encoder() -> HypergraphEncodings:
    """Create an encoder instance."""
    return HypergraphEncodings()


def test_ldp_encoding(
    sample_hypergraph: dict[str, dict[str, Any] | int | np.ndarray],
    encoder: HypergraphEncodings,
) -> None:
    """Test LDP encoding."""
    result = get_encodings(sample_hypergraph, encoder, "LDP")
    assert result is not None
    assert "features" in result
    assert isinstance(result["features"], np.ndarray)


def test_rwpe_encoding(
    sample_hypergraph: dict[str, dict[str, Any] | int | np.ndarray],
    encoder: HypergraphEncodings,
) -> None:
    """Test RWPE encoding."""
    k = 2
    result = get_encodings(sample_hypergraph, encoder, "RWPE", k_rwpe=k)
    assert result is not None
    assert "features" in result
    assert isinstance(result["features"], np.ndarray)
    assert result["features"].shape[1] == k + 1  # k + 1 features per node


def test_lcp_orc_encoding(
    sample_hypergraph: dict[str, dict[str, Any] | int | np.ndarray],
    encoder: HypergraphEncodings,
) -> None:
    """Test LCP-ORC encoding."""
    result = get_encodings(sample_hypergraph, encoder, "LCP-ORC")
    assert result is not None
    assert "features" in result
    assert isinstance(result["features"], np.ndarray)


def test_lape_normalized(
    sample_hypergraph: dict[str, dict[str, Any] | int | np.ndarray],
    encoder: HypergraphEncodings,
) -> None:
    """Test LAPE-Normalized encoding."""
    k = 2
    result = get_encodings(sample_hypergraph, encoder, "LAPE-Normalized", k_lape=k)
    assert result is not None
    assert "features" in result
    assert isinstance(result["features"], np.ndarray)
    assert (
        result["features"].shape[1] == k + 1
    )  # k + 1 features per node (includes original feature)


def test_invalid_encoding(
    sample_hypergraph: dict[str, dict[str, Any] | int | np.ndarray],
    encoder: HypergraphEncodings,
) -> None:
    """Test invalid encoding name."""
    result = get_encodings(sample_hypergraph, encoder, "INVALID")
    assert result is None


def test_copy_not_modified(
    sample_hypergraph: dict[str, dict[str, Any] | int | np.ndarray],
    encoder: HypergraphEncodings,
) -> None:
    """Test that original hypergraph is not modified."""
    original = sample_hypergraph.copy()
    _ = get_encodings(sample_hypergraph, encoder, "LDP")
    assert sample_hypergraph == original
