""" Test for the Laplacianss

Can use the toy hypergraph from our draft"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from encodings_hnns.laplacians import Laplacians


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


@pytest.fixture
def hodge_laplacian_up() -> np.ndarray:
    l_up: np.ndarray = np.array(
        [[3, 2, 1, 0], [2, 2, 1, 0], [1, 1, 3, 1], [0, 0, 1, 2]]
    )
    return l_up


@pytest.fixture
def hodge_laplacian_down() -> np.ndarray:
    l_down: np.ndarray = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 2, 2, 0, 0, 0],
            [1, 2, 3, 0, 1, 1],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 1, 1, 2, 1],
            [0, 0, 1, 0, 1, 1],
        ]
    )
    return l_down


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
    assert_array_equal(laplacian.boundary_matrix, boundary)


def test_compute_laplacian(toy_hypergraph, hodge_laplacian_down, hodge_laplacian_up):
    """

    Args:
        toy_hypergraph:
            hypergraph from draft
        laplacian_hodge:
            hodge laplacian (up)

    """
    laplacian: Laplacians = Laplacians(toy_hypergraph)
    # Computes the Forman-Ricci curvature
    laplacian.compute_hodge_laplacian()
    assert_array_equal(laplacian.hodge_laplacian_down, hodge_laplacian_down)
    assert_array_equal(laplacian.hodge_laplacian_up, hodge_laplacian_up)
