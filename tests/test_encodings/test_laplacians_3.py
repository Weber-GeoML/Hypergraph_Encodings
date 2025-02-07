"""Test for the Laplacianss

Can use the toy graph that is a star with node 1 connected to nodes 2, 3, 4, 5

I computed them by hand"""

from collections import OrderedDict

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from encodings_hnns.laplacians import Laplacians


@pytest.fixture
def toy_graph() -> dict[str, dict | int]:
    """Build toy hypergraph

    Returns:
        toy_graph:
            hypergraph from draft
    """
    # We don't care about features or labels
    hg: dict[str, dict | int] = {
        "hypergraph": {
            "yellow": [1, 2],
            "red": [1, 3],
            "blue": [1, 4],
            "green": [1, 5],
        },
        "features": {},
        "labels": {},
        "n": 3,
    }
    return hg


@pytest.fixture
def boundary() -> np.ndarray:
    """Returns the boundary matrix"""
    matrix_np: np.ndarray = np.array(
        [[1, 1, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )

    return matrix_np


@pytest.fixture
def degree_v() -> np.ndarray:
    """Returns the degree (vertices) matrix"""
    degree_vertices: np.ndarray = np.array(
        [
            [4, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )
    return degree_vertices


@pytest.fixture
def degree_v_inverse() -> np.ndarray:
    """Returns the inverse of the degree (vertices) matrix"""
    degree_vertices: np.ndarray = np.array(
        [
            [1 / 4, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )
    return degree_vertices


@pytest.fixture
def ngbors() -> OrderedDict:
    nhbors = OrderedDict(
        [
            (1, {1, 2, 3, 4, 5}),
            (2, {1, 2}),
            (3, {1, 3}),
            (4, {1, 4}),
            (5, {1, 5}),
        ]
    )
    return nhbors


@pytest.fixture
def degree_e() -> np.ndarray:
    """Returns the degree (edge) matrix"""
    d_e: np.ndarray = np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])
    return d_e


@pytest.fixture
def hodge_laplacian_up() -> np.ndarray:
    """Returns the Hodge Laplacian (down)"""
    l_up: np.ndarray = np.array(
        [[2, 1, 1, 1], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]]
    )
    return l_up


@pytest.fixture
def hodge_laplacian_down() -> np.ndarray:
    """Returns the Hodge Laplacian (up)"""
    l_down: np.ndarray = np.array(
        [
            [4, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1],
        ]
    )
    return l_down


@pytest.fixture
def normalized_laplacian() -> np.ndarray:
    """Returns the normalized Laplacian,
    which coincides with the simple graph Laplacian up to a factor of 1/2
    (Zhou: Learning with Hypergraphs: Clustering,
    Classification, and Embedding)"""
    # So I computed the simple graph Laplacian
    delta = np.array(
        [
            [1, -1 / 2, -1 / 2, -1 / 2, -1 / 2],
            [-1 / 2, 1, 0, 0, 0],
            [-1 / 2, 0, 1, 0, 0],
            [-1 / 2, 0, 0, 1, 0],
            [-1 / 2, 0, 0, 0, 1],
        ]
    )
    return (1 / 2) * delta


@pytest.fixture
def rw_laplacian() -> np.ndarray:
    """Returns the rw Laplacian, EE"""
    rw_laplacian: np.ndarray = np.array(
        [
            [1, -1 / 4, -1 / 4, -1 / 4, -1 / 4],
            [-1, 1, 0, 0, 0],
            [-1, 0, 1, 0, 0],
            [-1, 0, 0, 1, 0],
            [-1, 0, 0, 0, 1],
        ]
    )
    return rw_laplacian


@pytest.fixture
def rw_laplacian_en() -> np.ndarray:
    """Returns the rw Laplacian, EN"""
    l_en_alpha_0: np.ndarray = np.array(
        [
            [1, -1 / 4, -1 / 4, -1 / 4, -1 / 4],
            [-1, 1, 0, 0, 0],
            [-1, 0, 1, 0, 0],
            [-1, 0, 0, 1, 0],
            [-1, 0, 0, 0, 1],
        ]
    )
    return l_en_alpha_0


@pytest.fixture
def rw_laplacian_we() -> np.ndarray:
    """Returns the rw Laplacian, WE"""
    l_we_alpha_0 = np.array(
        [
            [1, -1 / 4, -1 / 4, -1 / 4, -1 / 4],
            [-1, 1, 0, 0, 0],
            [-1, 0, 1, 0, 0],
            [-1, 0, 0, 1, 0],
            [-1, 0, 0, 0, 1],
        ]
    )
    return l_we_alpha_0


def test_compute_boundary(toy_graph, boundary) -> None:
    """Test for compute_laplacian

    Args:
        toy_graph:
            hypergraph from draft
        boundary:
            the boundary matrix
    """
    laplacian: Laplacians = Laplacians(toy_graph)
    laplacian.compute_boundary()
    assert_array_equal(laplacian.boundary_matrix, boundary)


def test_compute_hodge_laplacian(
    toy_graph, hodge_laplacian_down, hodge_laplacian_up
) -> None:
    """Test for compute_hodge_laplacian

    Args:
        toy_graph:
            hypergraph from draft
        hodge_laplacian_down:
            hodge laplacian (down)
        hodge_laplacian_up:
            hodge laplacian (up)
    """
    laplacian: Laplacians = Laplacians(toy_graph)
    laplacian.compute_hodge_laplacian()
    assert_array_equal(laplacian.hodge_laplacian_down, hodge_laplacian_down)
    assert_array_equal(laplacian.hodge_laplacian_up, hodge_laplacian_up)


def test_compute_node_degree(toy_graph, degree_v) -> None:
    """Test for compute_edge_degree

    Args:
        toy_graph:
            hypergraph from draft
        degree_v:
            vertices degree matrix

    """
    laplacian: Laplacians = Laplacians(toy_graph)
    laplacian.compute_node_degrees()
    assert_array_equal(laplacian.degree_vertices, degree_v)


def test_compute_node_degree_bis(toy_graph, degree_v_inverse) -> None:
    """Test for compute_edge_degree

    Args:
        toy_graph:
            hypergraph from draft
        degree_v:
            vertices degree matrix

    """
    laplacian: Laplacians = Laplacians(toy_graph)
    laplacian.compute_node_degrees()
    assert_array_equal(np.linalg.inv(laplacian.degree_vertices), degree_v_inverse)


def test_compute_edge_degree(toy_graph, degree_e) -> None:
    """Test for compute_edge_degree

    Args:
        toy_graph:
            hypergraph from draft
        degree_e:
            edge degree matrix

    """
    laplacian: Laplacians = Laplacians(toy_graph)
    laplacian.compute_edge_degrees()
    assert_array_equal(laplacian.degree_edges, degree_e)


def test_compute_normalized_laplacian(toy_graph, normalized_laplacian) -> None:
    """Test for compute_normalized_laplacian

    Args:
        toy_graph:
            hypergraph from draft
        normalized_laplacian:
            normalized_laplacian

    """
    laplacian: Laplacians = Laplacians(toy_graph)
    laplacian.compute_normalized_laplacian()

    assert_allclose(
        laplacian.normalized_laplacian, normalized_laplacian, atol=1e-8
    ), f"The normalized laplacian is {laplacian.normalized_laplacian} and the expected is {normalized_laplacian}. The inverse of degree_vertices is {np.linalg.inv(laplacian.degree_vertices)}"


def test_compute_random_walk_laplacian_ee(toy_graph, rw_laplacian) -> None:
    """Test for compute_random_walk_laplacian (EE)

    Args:
        toy_graph:
            hypergraph from draft
        rw_laplacian:
            rw laplacian for EE scheme

    """
    laplacian: Laplacians = Laplacians(toy_graph)
    laplacian.compute_random_walk_laplacian(rw_type="EE")
    assert_allclose(laplacian.rw_laplacian, rw_laplacian, atol=1e-8)


def test_compute_node_neighbors(toy_graph, ngbors) -> None:
    """Test for compute_node_neighbors

    Args:
        toy_graph:
            hypergraph from draft
        nhbors:
            neighbors of each node in toy_graph

    """
    laplacian: Laplacians = Laplacians(toy_graph)
    laplacian.compute_node_neighbors(include_node=True)
    assert laplacian.node_neighbors == ngbors


def test_compute_random_walk_laplacian_en(toy_graph, rw_laplacian_en) -> None:
    """Test for compute_random_walk_laplacian (EN)

    Args:
        toy_graph:
            hypergraph from draft
        rw_laplacian_en:
            rw laplacian for EN scheme


    """
    laplacian: Laplacians = Laplacians(toy_graph)
    laplacian.compute_random_walk_laplacian(rw_type="EN")
    assert_allclose(laplacian.rw_laplacian, rw_laplacian_en, atol=1e-8)


def test_compute_random_walk_laplacian_we(toy_graph, rw_laplacian_we) -> None:
    """Test for compute_random_walk_laplacian (WE)

    Args:
        toy_graph:
            hypergraph from draft
        rw_laplacian_we:
            rw laplacian for WE scheme


    """
    laplacian: Laplacians = Laplacians(toy_graph)
    laplacian.compute_random_walk_laplacian(rw_type="WE")
    assert_allclose(laplacian.rw_laplacian, rw_laplacian_we, atol=1e-8)
