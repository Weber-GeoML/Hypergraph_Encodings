import pytest
import torch

from uniGCN.calculate_vertex_edges import calculate_v_e


@pytest.fixture
def simple_args():
    class Args:
        def __init__(self):
            self.add_self_loop = False
            self.first_aggregate = "mean"

    return Args()


@pytest.fixture
def simple_hypergraph():
    """
    Creates a simple hypergraph with 3 nodes and 2 edges:
    - Edge 0: {0, 1}
    - Edge 1: {1, 2}
    """
    X = torch.ones(3, 2)  # 3 nodes, 2 features each
    G = {
        "e0": [0, 1],  # First edge contains nodes 0 and 1
        "e1": [1, 2],  # Second edge contains nodes 1 and 2
    }
    return X, G


@pytest.fixture
def isolated_node_hypergraph():
    """
    Creates a hypergraph with an isolated node:
    - Edge 0: {0, 1}
    - Node 2: isolated
    """
    X = torch.ones(3, 2)
    G = {"e0": [0, 1]}  # Only one edge, node 2 is isolated
    return X, G


def test_basic_hypergraph(simple_args, simple_hypergraph):
    X, G = simple_hypergraph
    V, E, dege, degv, dege2 = calculate_v_e(X, G, simple_args)

    # Check shapes
    assert V.shape == E.shape  # Should have same number of non-zero elements
    assert len(dege) == 2  # Number of edges
    assert len(degv) == 3  # Number of nodes
    assert len(dege2) == 2  # Number of edges

    # Check specific values
    assert torch.allclose(degv.squeeze(), torch.tensor([1.0, 2.0, 1.0]).pow(-0.5))
    assert torch.all(~torch.isinf(degv))  # No infinite values
    assert torch.all(~torch.isnan(degv))  # No NaN values


def test_isolated_node(simple_args, isolated_node_hypergraph):
    X, G = isolated_node_hypergraph
    V, E, dege, degv, dege2 = calculate_v_e(X, G, simple_args)

    # Check that isolated node has degree 1 (after inf correction)
    assert degv[2] == 1.0  # Third node should have corrected degree

    # Check connectivity
    connected_nodes = torch.unique(V)
    assert 2 not in connected_nodes  # Node 2 should not appear in V


def test_degree_calculations(simple_args, simple_hypergraph):
    X, G = simple_hypergraph
    V, E, dege, degv, dege2 = calculate_v_e(X, G, simple_args)

    # Manual degree calculations for mean aggregation
    expected_degv = torch.tensor([[1.0], [2.0], [1.0]]).float().pow(-0.5)
    expected_dege = torch.tensor([[0.8165], [0.8165]])  # Updated expected values

    assert torch.allclose(degv, expected_degv)
    assert torch.allclose(dege, expected_dege, rtol=1e-3)


def test_sparse_matrix_conversion(simple_args, simple_hypergraph):
    X, G = simple_hypergraph
    V, E, dege, degv, dege2 = calculate_v_e(X, G, simple_args)

    # Check that V, E form a valid sparse representation
    assert torch.all(V >= 0)  # Valid node indices
    assert torch.all(V < X.shape[0])  # Within range
    assert torch.all(E >= 0)  # Valid edge indices
    assert torch.all(E < len(G))  # Within range


@pytest.mark.parametrize("aggregate_fn", ["mean", "sum", "min", "max"])
def test_different_aggregations(aggregate_fn, simple_hypergraph):
    class Args:
        def __init__(self):
            self.add_self_loop = False
            self.first_aggregate = aggregate_fn

    args = Args()
    X, G = simple_hypergraph

    # Should not raise error for different aggregation functions
    V, E, dege, degv, dege2 = calculate_v_e(X, G, args)
    assert not torch.any(torch.isnan(dege))
