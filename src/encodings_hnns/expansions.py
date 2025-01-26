import itertools

import hypernetx as hnx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# Compute the De matrices


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


def compute_star_expansion(dataset: dict) -> Data:
    """Compute the star expansion of a hypergraph.

    In star expansion, each hyperedge becomes a new vertex that connects to all vertices
    in that hyperedge, creating a bipartite graph.

    Args:
        dataset (dict): The dataset dictionary containing:
            - hypergraph: dict mapping hyperedge IDs to lists of node indices
            - n: number of original nodes
            - features: node features
            - labels: node labels

    Returns:
        Data: PyG Data object containing the expanded graph with:
            - Original node features for original nodes
            - Zero features for hyperedge nodes
            - Original labels for original nodes
            - Zero labels for hyperedge nodes
    """
    hypergraph = dataset["hypergraph"]
    num_orig_nodes = dataset["n"]

    # Create networkx graph for the bipartite expansion
    G = nx.Graph()

    # Add original nodes
    G.add_nodes_from(range(num_orig_nodes), bipartite=0)  # Original nodes

    # Add hyperedge nodes and their connections
    current_idx = num_orig_nodes
    for he_id, nodes in hypergraph.items():
        # Add new node for this hyperedge
        G.add_node(current_idx, bipartite=1)  # Hyperedge nodes

        # Connect to all nodes in the hyperedge
        G.add_edges_from([(current_idx, v) for v in nodes])
        current_idx += 1

    # Convert to PyG graph
    graph = from_networkx(G)

    # Handle node features
    num_he_nodes = len(hypergraph)
    orig_features = torch.tensor(dataset["features"]).float()
    feature_dim = orig_features.shape[1]

    # Create zero features for hyperedge nodes
    he_features = torch.zeros((num_he_nodes, feature_dim))

    # Combine original and hyperedge features
    graph.x = torch.cat([orig_features, he_features], dim=0)

    # Handle node labels similarly
    orig_labels = torch.tensor(dataset["labels"])
    he_labels = torch.zeros((num_he_nodes, orig_labels.shape[1]))
    graph.y = torch.cat([orig_labels, he_labels], dim=0)

    # Add a mask to identify original vs hyperedge nodes
    graph.original_mask = torch.cat(
        [
            torch.ones(num_orig_nodes, dtype=torch.bool),
            torch.zeros(num_he_nodes, dtype=torch.bool),
        ]
    )

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
