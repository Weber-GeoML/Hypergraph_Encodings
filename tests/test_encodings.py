""" Test for the curvature

Can use the toy hypergraph from our draft"""

import pytest
from numpy.testing import assert_array_equal

from encodings_hnns.encodings import HypergraphCurvatureProfile


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
def toy_hypergraph_2() -> dict[str, dict]:
    """Build toy hypergraph number 2

    Returns:
        toy_hypergraph_2:
            hypergraph
    """
    # We don't care about features or labels
    hg: dict[str, dict] = {
        "hypergraph": {
            "yellow": [4, 5, 7],
            "red": [5, 7],
        },
        "features": {},
        "labels": {},
        "n": 3,
    }
    return hg


@pytest.fixture
def hyperedges() -> dict[str, dict]:
    """Build toy hypergraph's hyperedges

    Returns:
        he:
            the dictionary with keys the nodes, values the
            hedges the node belongs to
    """
    # We don't care about features or labels
    he: dict[str, dict] = {
        1: ["yellow"],
        2: ["yellow", "red"],
        3: ["yellow", "red", "green"],
        4: ["blue"],
        5: ["green", "blue"],
        6: ["green"],
    }
    return he


@pytest.fixture
def hyperedges_2() -> dict[str, dict]:
    """Build toy hypergraph 2's hyperedges

    Returns:
        he:
            the dictionary with keys the nodes, values the
            hedges the node belongs to
    """
    # We don't care about features or labels
    he: dict[str, dict] = {
        4: ["yellow"],
        5: ["yellow", "red"],
        7: ["yellow", "red"],
    }
    return he


def test_compute_hyperedges(toy_hypergraph, hyperedges) -> None:
    """
    Test for compute_hyperedges

    Args:
        toy_hypergraph:
            hypergraph from draft
    """
    hypergraphcurvatureprofile: HypergraphCurvatureProfile = (
        HypergraphCurvatureProfile()
    )
    # Computes a dictionary called hyperedges.
    # The dictionary contains as keys the nodes,
    # as values the hyperedges the node belongs to.
    hypergraphcurvatureprofile.compute_hyperedges(toy_hypergraph)
    assert_array_equal(hypergraphcurvatureprofile.hyperedges, hyperedges)


def test_compute_hyperedges(toy_hypergraph_2, hyperedges_2) -> None:
    """
    Test for compute_hyperedges

    Args:
        toy_hypergraph_2:
            hypergraph number 2
    """
    hypergraphcurvatureprofile: HypergraphCurvatureProfile = (
        HypergraphCurvatureProfile()
    )
    # Computes a dictionary called hyperedges.
    # The dictionary contains as keys the nodes,
    # as values the hyperedges the node belongs to.
    hypergraphcurvatureprofile.compute_hyperedges(toy_hypergraph_2)
    assert_array_equal(hypergraphcurvatureprofile.hyperedges, hyperedges_2)
