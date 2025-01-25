import hypernetx as hnx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx
import itertools

# Compute the De matrices
from encodings_hnns.laplacians import Laplacians


def compute_clique_expansion(dataset: dict) -> Data:
    """Compute the clique expansion of a hypergraph.

    From Lukas

    Args:
        dataset (dict):
        The dataset dictionary containing the hypergraph, number of nodes, features, and labels.

    Returns:
        The expanded graph with node features and labels.
    """
    hypergraph = dataset["hypergraph"]
    num_nodes = dataset["n"]
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    # get edges
    hyperedges = list(hypergraph.values())
    for hyperedge in hyperedges:
        edges = itertools.combinations(hyperedge, 2)
        G.add_edges_from(edges)
    graph = from_networkx(G)

    # get node features
    graph.x = torch.tensor(dataset["features"]).float()

    # get node labels
    # print(dataset["labels"])
    # graph.y = torch.tensor([np.where(node == 1)[0][0] for node in dataset["labels"]])
    return graph


def plot_hypergraph_and_expansion() -> None:
    """Plot the hypergraph and its clique expansion on an example dataset."""
    # Create a simple example hypergraph
    hypergraph = {
        "e1": [0, 1, 2],  # A 3-node hyperedge
        "e2": [2, 3, 4],  # Another 3-node hyperedge
        "e3": [4, 5],  # A 2-node hyperedge
        "e4": [0, 5, 6, 7],  # A 4-node hyperedge
    }

    # Create dataset dictionary in the required format
    dataset = {
        "hypergraph": hypergraph,
        "n": 8,  # Number of nodes (0-7)
        "features": [[1.0] for _ in range(8)],  # Simple 1D features
        "labels": np.array(
            [[1] if i % 2 == 0 else [0] for i in range(8)]
        ),  # Binary labels as arrays
    }

    expanded_graph = compute_clique_expansion(dataset)
    G = nx.Graph()
    pos = nx.spring_layout(G)
    edge_index = expanded_graph.edge_index.t().tolist()
    G.add_edges_from(edge_index)

    # Plot original hypergraph
    plt.figure(figsize=(15, 6))

    # Hypergraph visualization
    plt.subplot(121)
    H = hnx.Hypergraph(hypergraph)
    pos = nx.spring_layout(H.bipartite())
    hnx.draw(H, with_node_labels=True, with_edge_labels=False, pos=pos)
    plt.title("Original Hypergraph", fontsize=20)

    # Compute and plot clique expansion
    plt.subplot(122)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        font_size=12,
        font_weight="bold",
    )
    plt.title("Clique Expansion", fontsize=20)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_hypergraph_and_expansion()
