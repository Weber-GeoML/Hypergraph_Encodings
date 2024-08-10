""" Test for the curvature

Can use the toy hypergraph from our draft"""

import pytest

from encodings_hnns.curvatures_frc import FormanRicci


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


def test_compute_node_degrees(toy_hypergraph) -> None:
    """
    Test for compute_node_degrees

    Args:
        toy_hypergraph:
            hypergraph from draft
    """
    forman_ricci = FormanRicci(toy_hypergraph)
    # Computes the Forman-Ricci curvature
    forman_ricci.compute_forman_ricci()
    print(forman_ricci.node_degrees)
    assert forman_ricci.node_degrees == {1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 1}


def test_compute_node_degrees_2(toy_hypergraph_2) -> None:
    """
    Test for compute_node_degrees

    Args:
        toy_hypergraph_2:
            hypergraph where the nodes do not start at 1
    """
    forman_ricci = FormanRicci(toy_hypergraph_2)
    # Computes the Forman-Ricci curvature
    forman_ricci.compute_forman_ricci()
    print(forman_ricci.node_degrees)
    assert forman_ricci.node_degrees == {4: 1, 5: 2, 7: 2}


def test_compute_forman_ricci(toy_hypergraph) -> None:
    """
    Test for compute_forman_ricci

    Args:
        toy_hypergraph:
            hypergraph from draft
    """
    forman_ricci: FormanRicci = FormanRicci(toy_hypergraph)
    # Computes the Forman-Ricci curvature
    forman_ricci.compute_forman_ricci()
    print(forman_ricci.forman_ricci)
    assert forman_ricci.forman_ricci == {
        "yellow": 0,
        "red": -1,
        "green": 0,
        "blue": 1,
    }
