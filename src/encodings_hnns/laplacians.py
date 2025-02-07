"""Add Laplacian encodings.

This module contains functions for computing the Laplacians
(Hodge, random walks, etc).
"""

from collections import OrderedDict

import numpy as np
from scipy.linalg import fractional_matrix_power


class DisconnectedError(Exception):
    """Exception raised when a node is found to be disconnected."""


class Laplacians:
    """Laplacian object."""
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
        # the node degrees
        self.node_degrees: dict = {}
        # the edge degrees
        self.edge_degrees: dict = {}
        self.node_neighbors: dict = {}
        # the boundary matrix
        self.boundary_matrix: None | np.ndarray = None
        self.normalized_laplacian: None | np.ndarray = None
        self.hodge_laplacian_up: None | np.ndarray = None
        self.hodge_laplacian_down: None | np.ndarray = None
        # the random walk laplacian
        self.rw_laplacian: None | np.ndarray = None
        # the local degree profile
        self.ldp: None | dict = None
        # the matrix of node/vertices degree
        self.degree_vertices: None | np.ndarray = None
        # the matrix of edge degrees
        self.degree_edges: None | np.ndarray = None
        # the ajacency matrix of a hypergraph is Defined in Zhou, Huang Scholkopf
        # as BB^T - D_v
        self.hypergraph_adjacency: None | np.ndarray = None

    def compute_boundary(self, verbose: bool = False) -> None:
        """Computes the boundary matrix.

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
            print(f"The nodes are {all_nodes}. \n We have {len(all_nodes)} nodes.")

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

    def compute_hypergraph_adjacency(self) -> None:
        """Computes the hypergraph adjacency matrix."""
        if self.boundary_matrix is None:
            self.compute_boundary()
        if self.degree_vertices is None:
            self.compute_node_degrees()

        assert self.boundary_matrix is not None
        assert self.degree_vertices is not None

        self.hypergraph_adjacency = (
            np.matmul(self.boundary_matrix, self.boundary_matrix.T)
            - self.degree_vertices
        )

    def compute_hodge_laplacian(self) -> None:
        """Computes the hodge-Laplacian of hypergraphs.

        There is the up Laplacian and the down Laplacian
        """
        if self.boundary_matrix is None:
            self.compute_boundary()
        assert self.boundary_matrix is not None
        self.hodge_laplacian_up = np.matmul(
            self.boundary_matrix.T, self.boundary_matrix
        )
        self.hodge_laplacian_down = np.matmul(
            self.boundary_matrix, self.boundary_matrix.T
        )

    def compute_normalized_laplacian(self) -> None:
        """Computes the normalized laplacian."""
        if self.boundary_matrix is None:
            # computes the boundary matrixs
            self.compute_boundary()
        if self.node_degrees == {}:
            # computes the node degrees
            self.compute_node_degrees()
        if not self.edge_degrees:
            # compute the edge degrees
            # that is number of nodes in each hyperedge
            self.compute_edge_degrees()

        assert self.boundary_matrix is not None
        assert self.degree_edges is not None
        assert self.degree_vertices is not None

        # intermediate is dv - B_1*D_e^{-1}*B_1^T
        assert self.boundary_matrix is not None
        assert self.degree_edges is not None
        intermediate = self.degree_vertices - np.matmul(
            self.boundary_matrix,
            np.matmul(np.linalg.inv(self.degree_edges), self.boundary_matrix.T),
        )

        dv_minus_onehalf = fractional_matrix_power(self.degree_vertices, -0.5)
        self.normalized_laplacian = np.matmul(
            dv_minus_onehalf, np.matmul(intermediate, dv_minus_onehalf)
        )

    def compute_random_walk_laplacian(
        self, verbose: bool = True, rw_type: str = "EN"
    ) -> None:
        """Computes the random walk Laplacian.

        Args:
            verbose:
                to print
            rw_type:
                EE, WE, EN (Equal Node)
        """
        if self.node_degrees == {}:
            self.compute_node_degrees()

        hypergraph: dict = self.hypergraph["hypergraph"]
        all_nodes: list = sorted(
            set(node for hyperedge in hypergraph.values() for node in hyperedge)
        )
        # NOTE: we noticed that the number of nodes is not the number of nodes reported in HyperGCN
        # so the following assert does not pass
        # assert len(all_nodes) == self.hypergraph["n"]
        if verbose:
            print(f"The nodes are {all_nodes}")

        # Initialize matrix (all zeros)
        num_nodes: int = len(all_nodes)
        matrix_ = np.zeros((num_nodes, num_nodes), dtype=float)

        if rw_type == "EE":  # equal edge. Banarjee
            # Here I am using the Approach i-D^{-1}A
            # Construct A_{ij} matrix
            # Iterate over node pairs (i, j) only for i < j to fill the upper triangular part
            # Takes advantage of symmetry (A is symmetric)
            for i, node_i in enumerate(all_nodes):
                for j in range(i + 1, num_nodes):
                    node_j = all_nodes[j]
                    for hyperedge in self.hypergraph["hypergraph"].values():
                        if set([node_i, node_j]).issubset(set(hyperedge)):
                            value = 1 / (len(hyperedge) - 1)
                            matrix_[i, j] += value
                            matrix_[
                                j, i
                            ] += value  # Reflect the value for symmetry (in A only!)

            # the rw Laplacian is I - dv^{-1}*A
            rw_l = np.eye(num_nodes) + np.matmul(
                np.linalg.inv(self.degree_vertices), -matrix_
            )
            # Note: this is not symmetric!
            self.rw_laplacian = rw_l

        if rw_type == "EN":  # equal node.
            # Note. Here I am not using D and A as above (ie I-D^{-1}A)
            # and as Raffaella's paper present the walks.
            if self.node_neighbors == {}:
                self.compute_node_neighbors()

            for i, node_i in enumerate(all_nodes):
                for j, node_j in enumerate(all_nodes):
                    if node_i == node_j:
                        matrix_[i, j] = 1
                    # go to any neighbor of with equal proba
                    elif node_j in self.node_neighbors[node_i]:
                        matrix_[i, j] = -1 / (len(self.node_neighbors[node_i]))

            self.rw_laplacian = matrix_

        if rw_type == "WE":  # weighted edge
            # Here is the idea: Sum the weights
            # of all hyperedges node i belongs to (|e|-1 if i is in e)
            # then for all j, if {i,j} belong to an edge, add 1 to count
            # then the value for matri_[i,j] is count/sum weights
            if self.node_neighbors == {}:
                self.compute_node_neighbors()

            for i, node_i in enumerate(all_nodes):
                i_neighbors_counts: dict = {}
                count_weights: int = 0
                for _, hedge in self.hypergraph["hypergraph"].items():
                    if node_i in hedge:
                        count_weights += len(hedge) - 1
                        for node_j in hedge:
                            if node_j not in i_neighbors_counts:
                                i_neighbors_counts[node_j] = 1
                            else:
                                i_neighbors_counts[node_j] += 1
                for j, node_j in enumerate(all_nodes):
                    if node_i == node_j:
                        matrix_[i, j] = 1  # diag
                    elif node_j not in self.node_neighbors[node_i]:
                        matrix_[i, j] = 0
                        matrix_[j, i] = (
                            0  # not neigbors so cannot travel between i and j
                        )
                    elif node_j in self.node_neighbors[node_i]:
                        matrix_[i, j] = -i_neighbors_counts[node_j] / count_weights
            self.rw_laplacian = matrix_

    def compute_ldp(self) -> None:
        """Computes the ldp. Local degree profile.

        Args:
            verbose:
                to print more info

        Outline:
            Iterate through each node.
            For each node, find its neighbors.
            Collect the degrees of these neighbors.
            Calculate the required statistics (min, max, median, and std) for these degrees.
            Store the results in a new dictionary.
        """

        if self.node_neighbors == {}:
            self.compute_node_neighbors()

        if self.node_degrees == {}:
            self.compute_node_degrees()

        result: dict = {}

        # loops through the nodes
        for node, neighbors in self.node_neighbors.items():
            # neighbors is the neighbors of node

            # Get the degrees of the neighbors
            neighbor_degrees = [self.node_degrees[neighbor] for neighbor in neighbors]
            assert neighbor_degrees

            # Calculate the statistics
            min_degree = np.min(neighbor_degrees)
            max_degree = np.max(neighbor_degrees)
            median_degree = np.median(neighbor_degrees)
            mean_degree = np.mean(neighbor_degrees)
            std_degree = np.std(neighbor_degrees)

            # Store the result
            result[node] = [
                self.node_degrees[node],
                min_degree,
                max_degree,
                median_degree,
                mean_degree,
                std_degree,
            ]

        self.ldp = result

    # This is also in curvatures so move this to parent class?
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

        self.degree_vertices = np.diag(list(self.node_degrees.values()))

    def compute_node_neighbors(self, include_node: bool = False) -> None:
        """Compute the neighbors of each node in the hypergraph.

        Args:
            inlcude_node:
                whether to inlcude node i in N(i).

        """
        assert self.node_neighbors == {}, "Node neighbors already computed."

        # loops though the hyperedges
        for hyperedge in self.hypergraph["hypergraph"].values():
            # loops through the nodes in the hyperedge.
            for node in hyperedge:
                if node not in self.node_neighbors:
                    self.node_neighbors[node] = set(hyperedge)
                else:
                    self.node_neighbors[node].update(hyperedge)
        # Sort the node degrees by keys
        self.node_neighbors = OrderedDict(sorted(self.node_neighbors.items()))

        if not include_node:
            # Remove i from the ith entry
            for i in self.node_neighbors.keys():
                self.node_neighbors[i].discard(i)

        # Assertion
        if not all(len(v) >= 1 for v in self.node_neighbors.values()):
            raise DisconnectedError(
                "A node is disconnected: one or more nodes have no neighbors."
            )

    def compute_edge_degrees(self) -> None:
        """Compute the degree of each hyperedge in the hypergraph."""
        assert not self.edge_degrees, "Edge degrees already computed."

        for hyperedge_name, hedge in self.hypergraph["hypergraph"].items():
            if hyperedge_name not in self.edge_degrees:
                self.edge_degrees[hyperedge_name] = len(hedge)

        self.degree_edges = np.diag(list(self.edge_degrees.values()))

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


# Example utilization
if __name__ == "__main__":

    print("EXAMPLE UTILIZATION")
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
    print(f"node_neighbors: \n {laplacian.node_neighbors}")
    laplacian.compute_node_degrees()
    print(f"node_degrees: \n {laplacian.node_degrees}")
    laplacian.compute_random_walk_laplacian(rw_type="WE")
    laplacian.compute_normalized_laplacian()
    laplacian.compute_random_walk_laplacian()
    print(f"rw_laplacian: \n {laplacian.rw_laplacian}")
    print("DONE")
