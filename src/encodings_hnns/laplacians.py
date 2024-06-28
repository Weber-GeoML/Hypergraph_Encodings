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
        self.node_neighbors: dict = {}
        self.boundary_matrix: None | np.ndarray = None
        self.normalized_laplacian: None | np.ndarray = None
        self.hodge_laplacian_up: None | np.ndarray = None
        self.hodge_laplacian_down: None | np.ndarray = None
        self.rw_laplacian: None | np.ndarray = None
        self.Dv: None | np.ndarray = None
        self.De: None | np.ndarray = None

    def compute_boundary(self, verbose: bool = False) -> None:
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
        """Computes the hodge-Laplacian of hypergraphs

        There is the up Laplacian and the down Laplacian
        """
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
        # Assumes that the nodes are sorted in increasing number

        intermediate = self.Dv - np.matmul(
            self.boundary_matrix,
            np.matmul(np.linalg.inv(self.De), self.boundary_matrix.T),
        )

        Dv_minus_onehalf = fractional_matrix_power(self.Dv, -0.5)
        self.normalized_laplacian = np.matmul(
            Dv_minus_onehalf, np.matmul(intermediate, Dv_minus_onehalf)
        )

    def compute_random_walk_laplacian(
        self, verbose: bool = True, type: str = "EE"
    ) -> None:
        """Computes the random walk Laplacian.

        Args:
            verbose:
                to print
            type:
                EE, WE, EN (Equal Node)
        """
        if self.node_degrees == {}:
            self.compute_node_degrees()

        hypergraph: dict = self.hypergraph["hypergraph"]
        all_nodes: list = sorted(
            set(node for hyperedge in hypergraph.values() for node in hyperedge)
        )
        if verbose:
            print(f"The nodes are {all_nodes}")

        # Initialize matrix
        num_nodes: int = len(all_nodes)
        matrix_ = np.zeros((num_nodes, num_nodes), dtype=float)

        if type == "EE":
            # Construct A_{ij} matrix
            # Iterate over node pairs (i, j) only for i < j to fill the upper triangular part
            # Takes advantage of symmetry
            for i, node_i in enumerate(all_nodes):
                for j in range(i + 1, num_nodes):
                    node_j = all_nodes[j]
                    for hyperedge in self.hypergraph["hypergraph"].values():
                        if set([node_i, node_j]).issubset(set(hyperedge)):
                            value = 1 / (len(hyperedge) - 1)
                            matrix_[i, j] += value
                            matrix_[j, i] += value  # Reflect the value for symmetry

            rw_l = np.eye(num_nodes) + np.matmul(np.linalg.inv(self.Dv), -matrix_)
            self.rw_laplacian = rw_l

        if type == "EN":
            # fucntion to get neighbors. Get len of neighbors of i
            pass
        if type == "WE":
            pass

    # def select nghbor_at_random( node):
    # edge_belong_to = []
    # for hyperedge in self.hypergraph["hypergraph"].values():
    #    if node in hyperedge:
    #       edge_belong_to.append(hyperedge)
    # then if EE
    # edge =random.choice(edge_belong_to)
    # edge.drop(node)
    # random.choice(edge)
    # if EN:
    # flattened_list = list(itertools.chain(*edge_belong_to))
    # edge.drop(node)
    # random.choice(edge)
    # if WE: to_do

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

    # def compute_node_neighbors(self) -> None:
    #     """Compute the neighbors of each node in the hypergraph."""
    #     assert self.node_neighbors == {}, "Node neighbors already computed."

    #     for hyperedge in self.hypergraph["hypergraph"].values():
    #         for node in hyperedge:
    #             if node not in self.node_neighbors:
    #                 self.node_neighbors[node] = hyperedge
    #             else:
    #                 self.node_neighbors[node].append(hyperedge)
    #     # Sort the node degrees by keys
    #     self.node_neighbors = OrderedDict(sorted(self.node_neighbors.items()))
    #     print(self.node_neighbors)
    #     assert False

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
    # print(data["features"])
    # print(data["labels"])

    # Instantiates the Laplacians class
    laplacian = Laplacians(data)
    laplacian.compute_node_neighbors()

    # Computes the Forman-Ricci curvature
    laplacian.compute_normalized_laplacian()
    laplacian.compute_random_walk_laplacian()
    print(laplacian.rw_laplacian)
