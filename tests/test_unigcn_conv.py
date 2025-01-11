import pytest
import torch
from uniGCN.UniGCN import UniGCNConv


@pytest.fixture
def simple_args():
    class Args:
        def __init__(self):
            self.add_self_loop = False
            self.first_aggregate = "mean"
            self.use_norm = True

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


def test_unigcn_conv_hypergraph_classification(simple_args, simple_hypergraph):
    X, G = simple_hypergraph

    # Calculate V, E from G
    vertex = []
    edges = []
    for edge_idx, (_, nodes) in enumerate(G.items()):
        vertex.extend(nodes)
        edges.extend([edge_idx] * len(nodes))
    vertex = torch.tensor(vertex)
    edges = torch.tensor(edges)

    conv = UniGCNConv(simple_args, in_channels=2, out_channels=2, heads=1)

    # Create degree tensors matching the structure from test_calculate_vertex_edges
    degE = torch.tensor([[0.8165], [0.8165]])  # From test_degree_calculations
    degV = torch.tensor([[1.0], [2.0], [1.0]]).pow(
        -0.5
    )  # From test_degree_calculations

    # Run forward pass with hypergraph classification
    output = conv(
        X, vertex, edges, hypergraph_classification=True, degE=degE, degV=degV
    )

    assert output.shape == (3, 2)
    assert not torch.isnan(output).any()
