"""
laplacians.py

This module contains functions for computing the Laplacians
(Hodge, random walks, etc).
"""

from collections import OrderedDict

import numpy as np
from scipy.linalg import fractional_matrix_power

from encodings_hnns.data_handling import parser


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
        self.node_degrees: dict = {}
        self.edge_degrees: dict = {}
        self.boundary_matrix: None | np.ndarray = None
        self.normalized_laplacian: None | np.ndarray = None
        self.hodge_laplacian_up: None | np.ndarray = None
        self.hodge_laplacian_down: None | np.ndarray = None
        self.Dv: None | np.ndarray = None
        self.De: None | np.ndarray = None

    def compute_boundary(self, verbose: bool = True) -> None:
        """Computes the boundary matrix

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

        self.boundary_matrix = matrix.T

    def compute_hodge_laplacian(self) -> None:
        """Computes the hodge-Laplacian of hypergraphs"""
        if self.boundary_matrix == None:
            self.compute_boundary()
        self.hodge_laplacian_up = np.matmul(
            self.boundary_matrix.T, self.boundary_matrix
        )
        self.hodge_laplacian_down = np.matmul(
            self.boundary_matrix, self.boundary_matrix.T
        )

    def compute_normalized_laplacian(self) -> None:
        """Computes the normalized laplacian"""
        if self.boundary_matrix == None:
            self.compute_boundary()
        if self.node_degrees == {}:
            self.compute_node_degrees()
        if self.edge_degrees == {}:
            self.compute_edge_degrees()
        # Assume that the nodes are sorted in increasing number? Should we do that?
        # Maybe that's the smart things to do?
        # Cause otherwise need the same order as the boundary matrix
        # TODO: make sure it's okay

        intermediate = self.Dv - np.matmul(
            self.boundary_matrix,
            np.matmul(np.linalg.inv(self.De), self.boundary_matrix.T),
        )

        Dv_minus_onehalf = fractional_matrix_power(self.Dv, -0.5)
        self.normalized_laplacian = np.matmul(
            Dv_minus_onehalf, np.matmul(intermediate, Dv_minus_onehalf)
        )

    def compute_random_walk_laplacian(self) -> None:
        pass

    # THis is also in curvatures so move this to parent class?
    def compute_node_degrees(self) -> None:
        """Compute the degree of each node in the hypergraph."""
        assert self.node_degrees == {}, "Node degrees already computed."

        for hyperedge in self.hypergraph["hypergraph"].values():
            for node in hyperedge:
                if node not in self.node_degrees:
                    self.node_degrees[node] = 1
                else:
                    self.node_degrees[node] += 1
        # Sort the node degrees by keys
        self.node_degrees = OrderedDict(sorted(self.node_degrees.items()))

        self.Dv: np.ndarray = np.diag(list(self.node_degrees.values()))

    def compute_edge_degrees(self) -> None:
        """Compute the degree of each hyperedge in the hypergraph."""
        assert self.edge_degrees == {}, "Edge degrees already computed."

        for hyperedge_name, hedge in self.hypergraph["hypergraph"].items():
            if hyperedge_name not in self.edge_degrees:
                self.edge_degrees[hyperedge_name] = len(hedge)

        self.De: np.ndarray = np.diag(list(self.edge_degrees.values()))


# Example utilization
if __name__ == "__main__":

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
    data = hg
    # print(data["hypergraph"])
    # So hypergraph is a dict:
    # key: authors, values: papers participates in.
    print(data["features"])
    print(data["labels"])

    # Instantiates the Laplacians class
    laplacian = Laplacians(data)

    # Computes the Forman-Ricci curvature
    laplacian.compute_normalized_laplacian()
    print(laplacian.De)
    print(laplacian.Dv)
    print(laplacian.normalized_laplacian)
