"""
laplacians.py

This module contains functions for computing the Laplacians
(Hodge, random walks, etc).
"""

import numpy as np


class Laplacians:
    def __init__(self, hypergraph) -> None:
        """Initialize the Laplacian object.

        Args:
            A hypergraph
            A hypergraph is a dictionary with the following keys:
                - hypergraph : a dictionary with the hyperedges as values
                - features : a dictionary with the features of the nodes as values
                - labels : a dictionary with the labels of the nodes as values
                - n : the number of nodes in the hypergraph
        """
        assert "hypergraph" in hypergraph.keys(), "Hypergraph not found."
        assert "features" in hypergraph.keys(), "Features not found."
        assert "labels" in hypergraph.keys(), "Labels not found."
        assert "n" in hypergraph.keys(), "Number of nodes not found."

        self.hypergraph = hypergraph
        self.node_degrees = {}
        self.boundary_matrix = None
        self.hodge_laplacian_up = None
        self.hodge_laplacian_down = None

    def compute_boundary(self, verbose: bool = True) -> None:
        """Computes the (Transpose) of the boundary matrix

        Args:
            verbose:
                to print more finely
        """
        # Extracts hypergraph and nodes
        hypergraph: dict = self.hypergraph["hypergraph"]
        all_nodes: list = sorted(
            set(node for hyperedge in hypergraph.values() for node in hyperedge)
        )
        if verbose:
            print(f"The nodes are {all_nodes}")

        # Creates mapping from nodes to column indices
        node_to_col: dict = {node: idx for idx, node in enumerate(all_nodes)}

        # Initialize matrix
        num_hyperedges: int = len(hypergraph)
        num_nodes: int = len(all_nodes)
        matrix = np.zeros((num_hyperedges, num_nodes), dtype=int)

        # Fill matrix based on hypergraph data
        for i, (hyperedge, nodes) in enumerate(hypergraph.items()):
            for node in nodes:
                j = node_to_col[node]
                matrix[i, j] = 1

        # this is actually the transpose technically
        self.boundary_matrix = matrix

    def compute_hodge_laplacian(self) -> None:
        """Computes the hodge-Laplacian of hypergraphs"""
        if self.boundary_matrix == None:
            self.compute_boundary()
        self.hodge_laplacian_up = np.matmul(
            self.boundary_matrix, self.boundary_matrix.T
        )
        self.hodge_laplacian_down = np.matmul(
            self.boundary_matrix.T, self.boundary_matrix
        )

    def compute_normalized_laplacian() -> None:
        pass

    def compute_random_walk_laplacian() -> None:
        pass

    def compute_node_degrees(self):
        """Compute the degree of each node in the hypergraph."""
        assert self.node_degrees == {}, "Node degrees already computed."

        for hyperedge in self.hypergraph["hypergraph"].values():
            for node in hyperedge:
                if node not in self.node_degrees:
                    self.node_degrees[node] = 1
                self.node_degrees[node] += 1
