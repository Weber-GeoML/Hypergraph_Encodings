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

    Same as 2 but different ordering
    To see if everything is fine.

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
def node_ldp() -> dict[int, list[float]]:
    """Returns the local degree profile for the toy hg"""
    # ordering is degree of node, min, max, median, mean, std
    # 1 has nbors 2, 3 with degree 2, 3
    # 2 has ngbors 1, 3 with degrees 1, 3
    # 3 has ngbors 1, 2, 5, 6 with degrees 1, 2, 2, 1
    # 4 has nghbor 5 with degree 2
    # 5 has nghbors 3, 4, 6 with deegree 3, 1, 1
    # 6 has nghbors 3, 5 with degrees 3, 2
    ldp: dict[int, list[float]] = {
        1: [1, 2, 3, 2.5, 2.5, 0.5],
        2: [2, 1, 3, 2, 2, 1],
        3: [3, 1, 2, 1.5, 1.5, 0.5],
        4: [1, 2, 2, 2, 2, 0],
        5: [2, 1, 3, 1, 5 / 3, np.std([1, 1, 3])],
        6: [1, 2, 3, 2.5, 2.5, 0.5],
    }

    return ldp


@pytest.fixture
def node_ldp_2() -> dict[int, list[float]]:
    """Returns the local degree profile for the toy hg 2"""
    # ordering is degree of node, min, max, median, mean, std
    # 4 has nbors 5,7 with degree 2, 2
    # 5 has ngbors 4, 7 with degrees 1, 2
    # 7 has ngbors 4, 5 with degrees 1, 2
    ldp: dict[int, list[float]] = {
        4: [1, 2, 2, 2, 2, 0],
        5: [2, 1, 2, 1.5, 1.5, 0.5],
        7: [2, 1, 2, 1.5, 1.5, 0.5],
    }
    return ldp


@pytest.fixture
def node_ldp_3() -> dict[int, list[float]]:
    """Returns the local degree profile for the toy hg 3"""
    # ordering is degree of node, min, max, median, mean, std
    # 4 has nbors 5,7 with degree 2, 2
    # 5 has ngbors 4, 7 with degrees 1, 2
    # 7 has ngbors 4, 5 with degrees 1, 2
    ldp: dict[int, list[float]] = {
        4: [1, 2, 2, 2, 2, 0],
        5: [2, 1, 2, 1.5, 1.5, 0.5],
        7: [2, 1, 2, 1.5, 1.5, 0.5],
    }
    return ldp


@pytest.fixture
def boundary() -> np.ndarray:
    """Returns the boundary matrix"""
    matrix_np: np.ndarray = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 1],
            [0, 0, 0, 1, 1, 0],
        ]
    )

    return matrix_np.T


@pytest.fixture
def boundary_2() -> np.ndarray:
    """Returns the boundary matrix for toy hg 2"""
    matrix_np: np.ndarray = np.array([[1, 0], [1, 1], [1, 1]])

    return matrix_np


@pytest.fixture
def boundary_3() -> np.ndarray:
    """Returns the boundary matrix for toy hg 3"""
    matrix_np: np.ndarray = np.array([[1, 0], [1, 1], [1, 1]])

    return matrix_np


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
def degree_v_2() -> np.ndarray:
    """Returns the degree (vertices) matrix"""
    D_v: np.ndarray = np.array(
        [
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
        ]
    )
    return D_v


@pytest.fixture
def degree_v_3() -> np.ndarray:
    """Returns the degree (vertices) matrix"""
    D_v: np.ndarray = np.array(
        [
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
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
def ngbors_2() -> np.ndarray:
    nhbors = OrderedDict(
        [
            (4, {4, 5, 7}),
            (5, {4, 5, 7}),
            (7, {4, 5, 7}),
        ]
    )
    return nhbors


@pytest.fixture
def ngbors_3() -> OrderedDict:
    nhbors = OrderedDict(
        [
            (4, {4, 5, 7}),
            (5, {4, 5, 7}),
            (7, {4, 5, 7}),
        ]
    )
    return nhbors


@pytest.fixture
def ngbors_not_inclusive() -> OrderedDict:
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
def ngbors_not_inclusive_2() -> OrderedDict:
    nhbors = OrderedDict(
        [
            (4, {5, 7}),
            (5, {4, 7}),
            (7, {4, 5}),
        ]
    )
    return nhbors


@pytest.fixture
def ngbors_not_inclusive_3() -> OrderedDict:
    nhbors = OrderedDict(
        [
            (4, {5, 7}),
            (5, {4, 7}),
            (7, {4, 5}),
        ]
    )
    return nhbors


@pytest.fixture
def degree_e() -> np.ndarray:
    """Returns the degree (edge) matrix"""
    d_e: np.ndarray = np.array([[3, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 2]])
    return d_e


@pytest.fixture
def degree_e_2() -> np.ndarray:
    """Returns the degree (edge) matrix for toy hg 2"""
    d_e: np.ndarray = np.array([[3, 0], [0, 2]])
    return d_e


@pytest.fixture
def degree_e_3() -> np.ndarray:
    """Returns the degree (edge) matrix for toy hg 3"""
    d_e: np.ndarray = np.array([[3, 0], [0, 2]])
    return d_e


@pytest.fixture
def hodge_laplacian_up() -> np.ndarray:
    """Returns the Hodge Laplacian (down)"""
    l_up: np.ndarray = np.array(
        [[3, 2, 1, 0], [2, 2, 1, 0], [1, 1, 3, 1], [0, 0, 1, 2]]
    )
    return l_up


@pytest.fixture
def hodge_laplacian_up_2() -> np.ndarray:
    """Returns the Hodge Laplacian (down)"""
    l_up: np.ndarray = np.array([[3, 2], [2, 2]])
    return l_up


@pytest.fixture
def hodge_laplacian_up_3() -> np.ndarray:
    """Returns the Hodge Laplacian (down)"""
    l_up: np.ndarray = np.array([[3, 2], [2, 2]])
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
def hodge_laplacian_down_2() -> np.ndarray:
    """Returns the Hodge Laplacian (up)"""
    l_down: np.ndarray = np.array(
        [
            [1, 1, 1],
            [1, 2, 2],
            [1, 2, 2],
        ]
    )
    return l_down


@pytest.fixture
def hodge_laplacian_down_3() -> np.ndarray:
    """Returns the Hodge Laplacian (up)"""
    l_down: np.ndarray = np.array(
        [
            [1, 1, 1],
            [1, 2, 2],
            [1, 2, 2],
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
def rw_laplacian_EN_2() -> np.ndarray:
    """Returns the rw Laplacian, EN"""
    rw_laplacian: np.ndarray = np.array(
        [
            [1, -1 / 2, -1 / 2],
            [-1 / 2, 1, -1 / 2],
            [-1 / 2, -1 / 2, 1],
        ]
    )
    return rw_laplacian


@pytest.fixture
def rw_laplacian_EN_3() -> np.ndarray:
    """Returns the rw Laplacian, EN"""
    rw_laplacian: np.ndarray = np.array(
        [
            [1, -1 / 2, -1 / 2],
            [-1 / 2, 1, -1 / 2],
            [-1 / 2, -1 / 2, 1],
        ]
    )
    return rw_laplacian


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


@pytest.fixture
def rw_laplacian_WE_2() -> np.ndarray:
    """Returns the rw Laplacian, WE"""
    L_WE_alpha_0 = np.array(
        [
            [1, -1 / 2, -1 / 2],
            [-1 / 3, 1, -2 / 3],
            [-1 / 3, -2 / 3, 1],
        ]
    )
    return L_WE_alpha_0


@pytest.fixture
def rw_laplacian_WE_3() -> np.ndarray:
    """Returns the rw Laplacian, WE"""
    L_WE_alpha_0 = np.array(
        [
            [1, -1 / 2, -1 / 2],
            [-1 / 3, 1, -2 / 3],
            [-1 / 3, -2 / 3, 1],
        ]
    )
    return L_WE_alpha_0


@pytest.fixture
def hypergraph_adjacency() -> np.ndarray:
    """Returns the hypergraph adjacency matrix for toy_hypergraph"""
    # For the hypergraph with edges:
    # yellow: [1, 2, 3]
    # red: [2, 3]
    # green: [3, 5, 6]
    # blue: [4, 5]
    # records how many edges are two pairs of nodes in together
    adj = np.array(
        [
            [0, 1, 1, 0, 0, 0],
            [1, 0, 2, 0, 0, 0],
            [1, 2, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, 0],
        ]
    )
    return adj


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


def test_compute_boundary_2(toy_hypergraph_2, boundary_2) -> None:
    """Test for compute_laplacian

    Args:
        toy_hypergraph_2:
            hypergraph from draft
        boundary_2:
            the boundary matrix
    """
    laplacian: Laplacians = Laplacians(toy_hypergraph_2)
    laplacian.compute_boundary()
    assert_array_equal(laplacian.boundary_matrix, boundary_2)


def test_compute_boundary_3(toy_hypergraph_3, boundary_3) -> None:
    """Test for compute_laplacian

    Args:
        toy_hypergraph_3:
            hypergraph from draft
        boundary_3:
            the boundary matrix
    """
    laplacian: Laplacians = Laplacians(toy_hypergraph_3)
    laplacian.compute_boundary()
    assert_array_equal(laplacian.boundary_matrix, boundary_3)


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


def test_compute_hodge_laplacian_2(
    toy_hypergraph_2, hodge_laplacian_down_2, hodge_laplacian_up_2
) -> None:
    """Test for compute_hodge_laplacian

    Args:
        toy_hypergraph_2:
            hypergraph from draft
        hodge_laplacian_down_2:
            hodge laplacian (down)
        hodge_laplacian_up_2:
            hodge laplacian (up)
    """
    laplacian: Laplacians = Laplacians(toy_hypergraph_2)
    laplacian.compute_hodge_laplacian()
    assert_array_equal(laplacian.hodge_laplacian_down, hodge_laplacian_down_2)
    assert_array_equal(laplacian.hodge_laplacian_up, hodge_laplacian_up_2)


def test_compute_hodge_laplacian_3(
    toy_hypergraph_3, hodge_laplacian_down_3, hodge_laplacian_up_3
) -> None:
    """Test for compute_hodge_laplacian

    Args:
        toy_hypergraph_3:
            hypergraph from draft
        hodge_laplacian_down_3:
            hodge laplacian (down)
        hodge_laplacian_up_3:
            hodge laplacian (up)
    """
    laplacian: Laplacians = Laplacians(toy_hypergraph_3)
    laplacian.compute_hodge_laplacian()
    assert_array_equal(laplacian.hodge_laplacian_down, hodge_laplacian_down_3)
    assert_array_equal(laplacian.hodge_laplacian_up, hodge_laplacian_up_3)


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
    assert_array_equal(laplacian.degree_vertices, degree_v)


def test_compute_node_degree_2(toy_hypergraph_2, degree_v_2) -> None:
    """Test for compute_edge_degree

    Args:
        toy_hypergraph_2:
            hypergraph from draft
        degree_v_2:
            vertices degree matrix

    """
    laplacian: Laplacians = Laplacians(toy_hypergraph_2)
    laplacian.compute_node_degrees()
    assert_array_equal(laplacian.degree_vertices, degree_v_2)


def test_compute_node_degree_3(toy_hypergraph_3, degree_v_3) -> None:
    """Test for compute_edge_degree

    Args:
        toy_hypergraph_3:
            hypergraph from draft
        degree_v_3:
            vertices degree matrix

    """
    laplacian: Laplacians = Laplacians(toy_hypergraph_3)
    laplacian.compute_node_degrees()
    assert_array_equal(laplacian.degree_vertices, degree_v_3)


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


def test_compute_ldp_2(toy_hypergraph_2, node_ldp_2) -> None:
    """Test for compute_ldp

    Args:
        toy_hypergraph_2:
            hypergraph
        node_ldp_2:
            the local degree profile

    """
    laplacian: Laplacians = Laplacians(toy_hypergraph_2)
    laplacian.compute_ldp()
    assert_array_equal(laplacian.ldp, node_ldp_2)


def test_compute_ldp_3(toy_hypergraph_3, node_ldp_3) -> None:
    """Test for compute_ldp

    Args:
        toy_hypergraph_3:
            hypergraph
        node_ldp_3:
            the local degree profile

    """
    laplacian: Laplacians = Laplacians(toy_hypergraph_3)
    laplacian.compute_ldp()
    assert_array_equal(laplacian.ldp, node_ldp_3)


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
    assert_array_equal(laplacian.degree_edges, degree_e)


def test_compute_edge_degree_2(toy_hypergraph_2, degree_e_2) -> None:
    """Test for compute_edge_degree

    Args:
        toy_hypergraph_2:
            hypergraph from draft
        degree_e_2:
            edge degree matrix

    """
    laplacian: Laplacians = Laplacians(toy_hypergraph_2)
    laplacian.compute_edge_degrees()
    assert_array_equal(laplacian.degree_edges, degree_e_2)


def test_compute_edge_degree_3(toy_hypergraph_3, degree_e_3) -> None:
    """Test for compute_edge_degree

    Args:
        toy_hypergraph_3:
            hypergraph from draft
        degree_e_3:
            edge degree matrix

    """
    laplacian: Laplacians = Laplacians(toy_hypergraph_3)
    laplacian.compute_edge_degrees()
    assert_array_equal(laplacian.degree_edges, degree_e_3)


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
    laplacian.compute_random_walk_laplacian(rw_type="EE")
    assert_allclose(laplacian.rw_laplacian, rw_laplacian, atol=1e-8)


def test_compute_node_neighbors(toy_hypergraph, ngbors) -> None:
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


def test_compute_node_neighbors_2(toy_hypergraph_2, ngbors_2) -> None:
    """Test for compute_node_neighbors

    Args:
        toy_hypergraph_2:
            hypergraph from draft
        nhbors_2:
            neighbors of each node in toy_hypergraph_2

    """
    laplacian: Laplacians = Laplacians(toy_hypergraph_2)
    laplacian.compute_node_neighbors(include_node=True)
    assert laplacian.node_neighbors == ngbors_2


def test_compute_node_neighbors_3(toy_hypergraph_3, ngbors_3) -> None:
    """Test for compute_node_neighbors

    Args:
        toy_hypergraph_3:
            hypergraph from draft
        nhbors_3:
            neighbors of each node in toy_hypergraph_3

    """
    laplacian: Laplacians = Laplacians(toy_hypergraph_3)
    laplacian.compute_node_neighbors(include_node=True)
    assert laplacian.node_neighbors == ngbors_3


def test_compute_node_neighbors_not_inclusive(
    toy_hypergraph, ngbors_not_inclusive
) -> None:
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


def test_compute_node_neighbors_not_inclusive_2(
    toy_hypergraph_2, ngbors_not_inclusive_2
) -> None:
    """Test for compute_node_neighbors

    Args:
        toy_hypergraph_2:
            hypergraph from draft
        nhbors_2:
            neighbors of each node in toy_hypergraph_2

    """
    laplacian: Laplacians = Laplacians(toy_hypergraph_2)
    laplacian.compute_node_neighbors(include_node=False)
    assert laplacian.node_neighbors == ngbors_not_inclusive_2


def test_compute_node_neighbors_not_inclusive_3(
    toy_hypergraph_3, ngbors_not_inclusive_3
) -> None:
    """Test for compute_node_neighbors

    Args:
        toy_hypergraph:
            hypergraph from draft
        nhbors:
            neighbors of each node in toy_hypergraph

    """
    laplacian: Laplacians = Laplacians(toy_hypergraph_3)
    laplacian.compute_node_neighbors(include_node=False)
    assert laplacian.node_neighbors == ngbors_not_inclusive_3


def test_compute_random_walk_laplacian_EN(toy_hypergraph, rw_laplacian_EN) -> None:
    """Test for compute_random_walk_laplacian (EN)

    Args:
        toy_hypergraph:
            hypergraph from draft
        rw_laplacian_EN:
            rw laplacian for EN scheme


    """
    laplacian: Laplacians = Laplacians(toy_hypergraph)
    laplacian.compute_random_walk_laplacian(rw_type="EN")
    assert_allclose(laplacian.rw_laplacian, rw_laplacian_EN, atol=1e-8)


def test_compute_random_walk_laplacian_EN_2(
    toy_hypergraph_2, rw_laplacian_EN_2
) -> None:
    """Test for compute_random_walk_laplacian (EN)

    Args:
        toy_hypergraph_2:
            hypergraph from draft
        rw_laplacian_EN_2:
            rw laplacian for EN scheme


    """
    laplacian: Laplacians = Laplacians(toy_hypergraph_2)
    laplacian.compute_random_walk_laplacian(rw_type="EN")
    assert_allclose(laplacian.rw_laplacian, rw_laplacian_EN_2, atol=1e-8)


def test_compute_random_walk_laplacian_EN_3(
    toy_hypergraph_3, rw_laplacian_EN_3
) -> None:
    """Test for compute_random_walk_laplacian (EN)

    Args:
        toy_hypergraph_3:
            hypergraph from draft
        rw_laplacian_EN_3:
            rw laplacian for EN scheme


    """
    laplacian: Laplacians = Laplacians(toy_hypergraph_3)
    laplacian.compute_random_walk_laplacian(rw_type="EN")
    assert_allclose(laplacian.rw_laplacian, rw_laplacian_EN_3, atol=1e-8)


def test_compute_random_walk_laplacian_WE(toy_hypergraph, rw_laplacian_WE) -> None:
    """Test for compute_random_walk_laplacian (WE)

    Args:
        toy_hypergraph:
            hypergraph from draft
        rw_laplacian_WE:
            rw laplacian for WE scheme


    """
    laplacian: Laplacians = Laplacians(toy_hypergraph)
    laplacian.compute_random_walk_laplacian(rw_type="WE")
    assert_allclose(laplacian.rw_laplacian, rw_laplacian_WE, atol=1e-8)


def test_compute_random_walk_laplacian_WE_2(
    toy_hypergraph_2, rw_laplacian_WE_2
) -> None:
    """Test for compute_random_walk_laplacian (WE)

    Args:
        toy_hypergraph_2:
            hypergraph from draft
        rw_laplacian_WE_2:
            rw laplacian for WE scheme


    """
    laplacian: Laplacians = Laplacians(toy_hypergraph_2)
    laplacian.compute_random_walk_laplacian(rw_type="WE")
    assert_allclose(laplacian.rw_laplacian, rw_laplacian_WE_2, atol=1e-8)


def test_compute_random_walk_laplacian_WE_3(
    toy_hypergraph_3, rw_laplacian_WE_3
) -> None:
    """Test for compute_random_walk_laplacian (WE)

    Args:
        toy_hypergraph_3:
            hypergraph from draft
        rw_laplacian_WE_3:
            rw laplacian for WE scheme


    """
    laplacian: Laplacians = Laplacians(toy_hypergraph_3)
    laplacian.compute_random_walk_laplacian(rw_type="WE")
    assert_allclose(laplacian.rw_laplacian, rw_laplacian_WE_3, atol=1e-8)


def test_compute_hypergraph_adjacency(toy_hypergraph, hypergraph_adjacency) -> None:
    """Test for compute_hypergraph_adjacency

    Args:
        toy_hypergraph:
            hypergraph from draft
        hypergraph_adjacency:
            expected adjacency matrix
    """
    laplacian: Laplacians = Laplacians(toy_hypergraph)
    laplacian.compute_hypergraph_adjacency()
    assert_array_equal(laplacian.hypergraph_adjacency, hypergraph_adjacency)
    laplacian.compute_hodge_laplacian()
    assert_array_equal(
        laplacian.hypergraph_adjacency,
        laplacian.hodge_laplacian_down - laplacian.degree_vertices,
    )
