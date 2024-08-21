""" Test for the curvature

Can use the toy hypergraph from our draft"""

# TODO: I would like to test if the encodings are empty
# And if there are already features

# NOTE: If we switch the order of the vertices, assume the order
# of the features are switched accordingly.

# TODO: test this way more, not just for ldp

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from encodings_hnns.encodings import HypergraphEncodings


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
def toy_hypergraph_with_features() -> dict[str, dict]:
    """Build toy hypergraph with features

    Returns:
        toy_hypergraph:
            hypergraph from draft
    """
    # We don't care about features or labels
    # CHANGE THE VERTICES: SUBSTRACTED -1
    hg: dict[str, dict] = {
        "hypergraph": {
            "yellow": [0, 1, 2],
            "red": [1, 2],
            "green": [2, 4, 5],
            "blue": [3, 4],
        },
        "features": np.matrix([[1], [1], [1], [1], [1], [1]]),
        "labels": {},
        "n": 6,
    }
    return hg


@pytest.fixture
def toy_hypergraph_with_encodings_ldp() -> dict[str, dict]:
    """Build toy hypergraph with features + encodings

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
        # features added from LDP
        "features": np.matrix(
            [
                [1, 1, 2, 3, 2.5, 2.5, 0.5],
                [1, 2, 1, 3, 2, 2, 1],
                [1, 3, 1, 2, 1.5, 1.5, 0.5],
                [1, 1, 2, 2, 2, 2, 0],
                [1, 2, 1, 3, 1, 5 / 3, np.std([1, 1, 3])],
                [1, 1, 2, 3, 2.5, 2.5, 0.5],
            ]
        ),
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


# NOTE: We mandate that the hyperedges are sorted
# This is a fixture to see if our code is robust
# even if one does not follow the aforementioned guideline
@pytest.fixture
def toy_hypergraph_3() -> dict[str, dict]:
    """Build toy hypergraph number 3

    Returns:
        toy_hypergraph_3:
            hypergraph
    """
    # We don't care about features or labels
    hg: dict[str, dict] = {
        "hypergraph": {
            "yellow": [7, 5, 4],
            "red": [7, 5],
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


@pytest.fixture
def hyperedges_3() -> dict[str, dict]:
    """Build toy hypergraph 3's hyperedges

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


def test_add_degree_encodings(
    toy_hypergraph_with_features, toy_hypergraph_with_encodings_ldp
) -> None:
    """
    Test for add_degree_encodings

    Args:
        toy_hypergraph:
            hypergraph from draft
    """
    hgencodings: HypergraphEncodings = HypergraphEncodings()
    toy_hypergraph_with_features = hgencodings.add_degree_encodings(
        toy_hypergraph_with_features
    )
    assert_array_equal(
        toy_hypergraph_with_features["features"],
        toy_hypergraph_with_encodings_ldp["features"],
    )


# def test_add_lapacian_encodings(toy_hypergraph_with_features) -> None:
#     """
#     Test for add_degree_encodings

#     Args:
#         toy_hypergraph:
#             hypergraph from draft
#     """
#     hgencodings: HypergraphEncodings = HypergraphEncodings()
#     toy_hypergraph_with_features = hgencodings.add_laplacian_encodings(
#         toy_hypergraph_with_features
#     )
#     assert_array_equal(
#         toy_hypergraph_with_features["features"].shape[0] == 6
#     ), f"the shape is {toy_hypergraph_with_features['features'].shape[0]}"


def test_compute_hyperedges(toy_hypergraph, hyperedges) -> None:
    """
    Test for compute_hyperedges

    Args:
        toy_hypergraph:
            hypergraph from draft
        hyperedges:
            the dict of keys: nodes, values: hyperedges
    """
    hgencodings: HypergraphEncodings = HypergraphEncodings()
    # Computes a dictionary called hyperedges.
    # The dictionary contains as keys the nodes,
    # as values the hyperedges the node belongs to.
    hgencodings.compute_hyperedges(toy_hypergraph)
    assert_array_equal(hgencodings.hyperedges, hyperedges)


def test_compute_hyperedges_2(toy_hypergraph_2, hyperedges_2) -> None:
    """
    Test for compute_hyperedges

    Args:
        toy_hypergraph_2:
            hypergraph number 2
        hyperedges_2:
            the dict of keys: nodes, values: hyperedges
            for hg 2
    """
    hgencodings: HypergraphEncodings = HypergraphEncodings()
    # Computes a dictionary called hyperedges.
    # The dictionary contains as keys the nodes,
    # as values the hyperedges the node belongs to.
    hgencodings.compute_hyperedges(toy_hypergraph_2)
    assert_array_equal(hgencodings.hyperedges, hyperedges_2)


def test_compute_hyperedges_3(toy_hypergraph_3, hyperedges_3) -> None:
    """
    Test for compute_hyperedges

    Args:
        toy_hypergraph_3:
            hypergraph number 3
        hyperedges_3:
            the dict of keys: nodes, values: hyperedges
            for hg 3
    """
    hgencodings: HypergraphEncodings = HypergraphEncodings()
    # Computes a dictionary called hyperedges.
    # The dictionary contains as keys the nodes,
    # as values the hyperedges the node belongs to.
    hgencodings.compute_hyperedges(toy_hypergraph_3)
    assert_array_equal(hgencodings.hyperedges, hyperedges_3)
