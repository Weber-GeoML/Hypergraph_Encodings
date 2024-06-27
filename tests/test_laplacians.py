""" Test for the Laplacianss

Can use the toy hypergraph from our draft"""

import pytest
from encodings_hnns.laplacians import Laplacians
import numpy as np
from numpy.testing import assert_array_equal


@pytest.fixture
def toy_hypergraph() -> dict[str, dict]:
    """Build toy hypergraph

    Returns:
        toy_hypergraph:
            hypergraph from draft
    """
    # We don't care about features or labels
    hg: dict[str, dict] = {
        "hypergraph": {
            "yellow": [1, 2, 3],
            "red": [2, 3],
            "green": [3, 5, 6],
            "blue": [4, 5],
        },
        "features": {},
        "labels": {},
        "n": 6,
    }
    return hg


@pytest.fixture
def boundary() -> np.ndarray:
    matrix_np: np.ndarray = np.array(
        [[1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 1, 0]]
    )

    return matrix_np


def test_compute_boundary(toy_hypergraph, boundary) -> None:
    """
    Test for compute_laplacian

    Args:
        toy_hypergraph:
            hypergraph from draft
    """
    laplacian: Laplacians = Laplacians(toy_hypergraph)
    # Computes the Forman-Ricci curvature
    laplacian.compute_boundary()
    print(laplacian.boundary_matrix)
    assert_array_equal(laplacian.boundary_matrix, boundary)
