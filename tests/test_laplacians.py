""" Test for the Laplacianss

Can use the toy hypergraph from our draft"""

from collections import OrderedDict

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from encodings_hnns.laplacians import Laplacians


@pytest.fixture
def toy_hypergraph() -> dict[str, dict | int]:
    """Build toy hypergraph

    Returns:
        toy_hypergraph:
            hypergraph from draft
    """
    # We don't care about features or labels
    hg: dict[str, dict | int] = {
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
def node_ldp() -> np.ndarray:
    """Returns the local degree profile for the toy hg"""
    # ordering is degree of node, min, max, median, mean, std
    # 1 has nbors 2, 3 with degree 2, 3
    # 2 has ngbors 1, 3 with degrees 1, 3
    # 3 has ngbors 1, 2, 5, 6 with degrees 1, 2, 2, 1
    # 4 has nghbor 5 with degree 2
    # 5 has nghbors 3, 4, 6 with deegree 3, 1, 1
    # 6 has nghbors 3, 5 with degrees 3, 2
    ldp: np.ndarray = {
        1: [1, 2, 3, 2.5, 2.5, 0.5],
        2: [2, 1, 3, 2, 2, 1],
        3: [3, 1, 2, 1.5, 1.5, 0.5],
        4: [1, 2, 2, 2, 2, 0],
        5: [2, 1, 3, 1, 5 / 3, np.std([1, 1, 3])],
        6: [1, 2, 3, 2.5, 2.5, 0.5],
    }

    return ldp


@pytest.fixture
def boundary() -> np.ndarray:
    """Returns the boundary matrix"""
    matrix_np: np.ndarray = np.array(
        [[1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 1, 0]]
    )

    return matrix_np.T


@pytest.fixture
def degree_v() -> np.ndarray:
    """Returns the degree (vertices) matrix"""
    D_v: np.ndarray = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )
    return D_v


@pytest.fixture
def ngbors() -> np.ndarray:
    nhbors = OrderedDict(
        [
            (1, {1, 2, 3}),
            (2, {1, 2, 3}),
            (3, {1, 2, 3, 5, 6}),
            (4, {4, 5}),
            (5, {3, 4, 5, 6}),
            (6, {3, 5, 6}),
        ]
    )
    return nhbors


@pytest.fixture
def ngbors_not_inclusive() -> np.ndarray:
    nhbors = OrderedDict(
        [
            (1, {2, 3}),
            (2, {1, 3}),
            (3, {1, 2, 5, 6}),
            (4, {5}),
            (5, {3, 4, 6}),
            (6, {3, 5}),
        ]
    )
    return nhbors


@pytest.fixture
def degree_e() -> np.ndarray:
    """Returns the degree (edge) matrix"""
    d_e: np.ndarray = np.array([[3, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 2]])
    return d_e


@pytest.fixture
def hodge_laplacian_up() -> np.ndarray:
    """Returns the Hodge Laplacian (down)"""
    l_up: np.ndarray = np.array(
        [[3, 2, 1, 0], [2, 2, 1, 0], [1, 1, 3, 1], [0, 0, 1, 2]]
    )
    return l_up


@pytest.fixture
def hodge_laplacian_down() -> np.ndarray:
    """Returns the Hodge Laplacian (up)"""
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


@pytest.fixture
def normalized_laplacian() -> np.ndarray:
    """Returns the normalized Laplacian"""
    Delta = np.array(
        [
            [2 / 3, -1 / (3 * np.sqrt(2)), -1 / (3 * np.sqrt(3)), 0, 0, 0],
            [-1 / (3 * np.sqrt(2)), 7 / 12, -5 / (6 * np.sqrt(6)), 0, 0, 0],
            [
                -1 / (3 * np.sqrt(3)),
                -5 / (6 * np.sqrt(6)),
                11 / 18,
                0,
                -1 / (3 * np.sqrt(6)),
                -1 / (3 * np.sqrt(3)),
            ],
            [0, 0, 0, 1 / 2, -1 / (2 * np.sqrt(2)), 0],
            [
                0,
                0,
                -1 / (3 * np.sqrt(6)),
                -1 / (2 * np.sqrt(2)),
                7 / 12,
                -1 / (3 * np.sqrt(2)),
            ],
            [0, 0, -1 / (3 * np.sqrt(3)), 0, -1 / (3 * np.sqrt(2)), 2 / 3],
        ]
    )
    return Delta


@pytest.fixture
def rw_laplacian() -> np.ndarray:
    """Returns the rw Laplacian, EE"""
    rw_laplacian: np.ndarray = np.array(
        [
            [1, -1 / 2, -1 / 2, 0, 0, 0],
            [-1 / 4, 1, -3 / 4, 0, 0, 0],
            [-1 / 6, -1 / 2, 1, 0, -1 / 6, -1 / 6],
            [0, 0, 0, 1, -1, 0],
            [0, 0, -1 / 4, -1 / 2, 1, -1 / 4],
            [0, 0, -1 / 2, 0, -1 / 2, 1],
        ]
    )
    return rw_laplacian


@pytest.fixture
def rw_laplacian_EN() -> np.ndarray:
    """Returns the rw Laplacian, EN"""
    L_EN_alpha_0: np.ndarray = np.array(
        [
            [1, -1 / 2, -1 / 2, 0, 0, 0],
            [-1 / 2, 1, -1 / 2, 0, 0, 0],
            [-1 / 4, -1 / 4, 1, 0, -1 / 4, -1 / 4],
            [0, 0, 0, 1, -1, 0],
            [0, 0, -1 / 3, -1 / 3, 1, -1 / 3],
            [0, 0, -1 / 2, 0, -1 / 2, 1],
        ]
    )
    return L_EN_alpha_0


@pytest.fixture
def rw_laplacian_WE() -> np.ndarray:
    """Returns the rw Laplacian, WE"""
    L_WE_alpha_0 = np.array(
        [
            [1, -1 / 2, -1 / 2, 0, 0, 0],
            [-1 / 3, 1, -2 / 3, 0, 0, 0],
            [-1 / 5, -2 / 5, 1, 0, -1 / 5, -1 / 5],
            [0, 0, 0, 1, -1, 0],
            [0, 0, -1 / 3, -1 / 3, 1, -1 / 3],
            [0, 0, -1 / 2, 0, -1 / 2, 1],
        ]
    )
    return L_WE_alpha_0


def test_compute_boundary(toy_hypergraph, boundary) -> None:
    """Test for compute_laplacian

    Args:
        toy_hypergraph:
            hypergraph from draft
        boundary:
            the boundary matrix
    """
    laplacian: Laplacians = Laplacians(toy_hypergraph)
    laplacian.compute_boundary()
    assert_array_equal(laplacian.boundary_matrix, boundary)


def test_compute_hodge_laplacian(
    toy_hypergraph, hodge_laplacian_down, hodge_laplacian_up
) -> None:
    """Test for compute_hodge_laplacian

    Args:
        toy_hypergraph:
            hypergraph from draft
        hodge_laplacian_down:
            hodge laplacian (down)
        hodge_laplacian_up:
            hodge laplacian (up)
    """
    laplacian: Laplacians = Laplacians(toy_hypergraph)
    laplacian.compute_hodge_laplacian()
    assert_array_equal(laplacian.hodge_laplacian_down, hodge_laplacian_down)
    assert_array_equal(laplacian.hodge_laplacian_up, hodge_laplacian_up)


def test_compute_node_degree(toy_hypergraph, degree_v) -> None:
    """Test for compute_edge_degree

    Args:
        toy_hypergraph:
            hypergraph from draft
        degree_v:
            vertices degree matrix

    """
    laplacian: Laplacians = Laplacians(toy_hypergraph)
    laplacian.compute_node_degrees()
    assert_array_equal(laplacian.Dv, degree_v)


def test_compute_ldp(toy_hypergraph, node_ldp) -> None:
    """Test for compute_ldp

    Args:
        toy_hypergraph:
            hypergraph from draft
        node_ldp:
            the local degree profile

    """
    laplacian: Laplacians = Laplacians(toy_hypergraph)
    laplacian.compute_ldp()
    assert_array_equal(laplacian.ldp, node_ldp)


def test_compute_edge_degree(toy_hypergraph, degree_e) -> None:
    """Test for compute_edge_degree

    Args:
        toy_hypergraph:
            hypergraph from draft
        degree_e:
            edge degree matrix

    """
    laplacian: Laplacians = Laplacians(toy_hypergraph)
    laplacian.compute_edge_degrees()
    assert_array_equal(laplacian.De, degree_e)


def test_compute_normalized_laplacian(toy_hypergraph, normalized_laplacian) -> None:
    """Test for compute_normalized_laplacian

    Args:
        toy_hypergraph:
            hypergraph from draft
        normalized_laplacian:
            normalized_laplacian

    """
    laplacian: Laplacians = Laplacians(toy_hypergraph)
    laplacian.compute_normalized_laplacian()
    assert_allclose(laplacian.normalized_laplacian, normalized_laplacian, atol=1e-8)


def test_compute_random_walk_laplacian_EE(toy_hypergraph, rw_laplacian) -> None:
    """Test for compute_random_walk_laplacian (EE)

    Args:
        toy_hypergraph:
            hypergraph from draft
        rw_laplacian:
            rw laplacian for EE scheme

    """
    laplacian: Laplacians = Laplacians(toy_hypergraph)
    laplacian.compute_random_walk_laplacian(type="EE")
    assert_allclose(laplacian.rw_laplacian, rw_laplacian, atol=1e-8)


def test_compute_node_neighbors(toy_hypergraph, ngbors):
    """Test for compute_node_neighbors

    Args:
        toy_hypergraph:
            hypergraph from draft
        nhbors:
            neighbors of each node in toy_hypergraph

    """
    laplacian: Laplacians = Laplacians(toy_hypergraph)
    laplacian.compute_node_neighbors(include_node=True)
    assert laplacian.node_neighbors == ngbors


def test_compute_node_neighbors_not_inclusive(toy_hypergraph, ngbors_not_inclusive):
    """Test for compute_node_neighbors

    Args:
        toy_hypergraph:
            hypergraph from draft
        nhbors:
            neighbors of each node in toy_hypergraph

    """
    laplacian: Laplacians = Laplacians(toy_hypergraph)
    laplacian.compute_node_neighbors(include_node=False)
    assert laplacian.node_neighbors == ngbors_not_inclusive


def test_compute_random_walk_laplacian_EN(toy_hypergraph, rw_laplacian_EN) -> None:
    """Test for compute_random_walk_laplacian (EN)

    Args:
        toy_hypergraph:
            hypergraph from draft
        rw_laplacian_EN:
            rw laplacian for EN scheme


    """
    laplacian: Laplacians = Laplacians(toy_hypergraph)
    laplacian.compute_random_walk_laplacian(type="EN")
    assert_allclose(laplacian.rw_laplacian, rw_laplacian_EN, atol=1e-8)


def test_compute_random_walk_laplacian_WE(toy_hypergraph, rw_laplacian_WE) -> None:
    """Test for compute_random_walk_laplacian (WE)

    Args:
        toy_hypergraph:
            hypergraph from draft
        rw_laplacian_WE:
            rw laplacian for WE scheme


    """
    laplacian: Laplacians = Laplacians(toy_hypergraph)
    laplacian.compute_random_walk_laplacian(type="WE")
    assert_allclose(laplacian.rw_laplacian, rw_laplacian_WE, atol=1e-8)
