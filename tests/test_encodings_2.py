""" Test for the curvature

Can use the toy hypergraph from our draft"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from encodings_hnns.encodings import HypergraphEncodings


import csv

# Initialize an empty dictionary to store the loaded data
loaded_hypergraph = {}

# Open the CSV file and read it
with open("debug.csv", "r", newline="") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        key = row[0]
        # If the row has more than one element, assume the remaining are the values
        if len(row) > 2:
            value = row[1:]
        else:
            value = row[1]
        loaded_hypergraph[key] = value


@pytest.fixture
def toy_hypergraph() -> dict[str, dict]:
    """Build toy hypergraph

    Returns:
        toy_hypergraph:
            hypergraph
    """
    # We don't care about features or labels
    hg: dict[str, dict] = {
        "hypergraph": {
            1: [1, 2, 3, 7, 8, 9, 10, 11],
            2: [2, 3],
            3: [3, 5, 6],
            4: [4, 5],
        },
        "features": {},
        "labels": {},
        "n": 6,
    }
    return hg


@pytest.fixture
def toy_hypergraph_2() -> dict[str, dict]:
    """Build toy hypergraph

    Returns:
        toy_hypergraph:
            hypergraph
    """
    # We don't care about features or labels
    hg: dict[str, dict] = {
        "hypergraph": {
            1: [1, 2, 3, 7, 8, 9, 10, 11],
        },
        "features": {},
        "labels": {},
        "n": 6,
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
        1: [1],
        2: [1, 2],
        3: [1, 2, 3],
        4: [4],
        5: [3, 4],
        6: [3],
        7: [1],
        8: [1],
        9: [1],
        10: [1],
        11: [1],
    }
    return he


@pytest.fixture
def hyperedges_2() -> dict[str, dict]:
    """Build toy hypergraph's hyperedges

    Returns:
        he:
            the dictionary with keys the nodes, values the
            hedges the node belongs to
    """
    # We don't care about features or labels
    he: dict[str, dict] = {
        1: [1],
        2: [1],
        3: [1],
        7: [1],
        8: [1],
        9: [1],
        10: [1],
        11: [1],
    }
    return he


def test_compute_hyperedges(toy_hypergraph, hyperedges) -> None:
    """
    Test for compute_hyperedges

    Args:
        toy_hypergraph:
            hypergraph
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
            hypergraph
        hyperedges_2:
            the dict of keys: nodes, values: hyperedges
    """
    hgencodings: HypergraphEncodings = HypergraphEncodings()
    # Computes a dictionary called hyperedges.
    # The dictionary contains as keys the nodes,
    # as values the hyperedges the node belongs to.
    hgencodings.compute_hyperedges(toy_hypergraph_2)
    assert_array_equal(hgencodings.hyperedges, hyperedges_2)
