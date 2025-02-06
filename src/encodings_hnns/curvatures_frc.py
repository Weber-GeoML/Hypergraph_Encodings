"""
curvatures_frc.py

This module contains functions for computing the Forman-Ricci 
curvatures of an undirected hypergraph.
"""

import sys

from encodings_hnns.data_handling import parser

print(sys.path)


class FormanRicci:
    def __init__(self, hypergraph: dict) -> None:
        """Initialize the Forman-Ricci curvature object.

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

        self.hypergraph: dict = hypergraph
        self.node_degrees: dict = {}
        self.forman_ricci: dict = {}

    def compute_forman_ricci(self) -> None:
        """Compute the Forman-Ricci curvature of a hyperedge.

        The formula is:
        F(e) = 2|e| - D
        where |e| is the number of nodes in the hyperedge and D is the sum of the degrees.
        """
        assert (
            self.forman_ricci == {}
        ), "Forman-Ricci curvature already computed."

        if self.node_degrees == {}:
            self.compute_node_degrees()

        # for hyperedge in self.hypergraph["hypergraph"].values():
        #     hyperedge_degree = len(hyperedge)
        #     hyperedge_sum_degrees = sum([self.node_degrees[node] for node in hyperedge])
        #     tuple_hyperedge = tuple(hyperedge)
        #     self.forman_ricci[tuple_hyperedge] = (
        #         2 * hyperedge_degree - hyperedge_sum_degrees
        #     )
        # Lukas: why not take advantage of the name of the hyper-edges?
        for name, hyperedge in self.hypergraph["hypergraph"].items():
            hyperedge_degree = len(hyperedge)
            hyperedge_sum_degrees = sum(
                [self.node_degrees[node] for node in hyperedge]
            )
            self.forman_ricci[name] = (
                2 * hyperedge_degree - hyperedge_sum_degrees
            )

    def compute_node_degrees(self) -> None:
        """Compute the degree of each node in the hypergraph."""
        assert self.node_degrees == {}, "Node degrees already computed."

        for hyperedge in self.hypergraph["hypergraph"].values():
            for node in hyperedge:
                if node not in self.node_degrees:
                    self.node_degrees[node] = 1
                else:
                    self.node_degrees[node] += 1


# Example utilization
if __name__ == "__main__":
    print("EXAMPLE UTILIZATION")
    data_type = "coauthorship"
    dataset_name = "cora"
    # Create an instance of the parser class
    parser_instance = parser(data_type, dataset_name)
    data = parser_instance._load_data()
    # print(data["hypergraph"])
    # So hypergraph is a dict:
    # key: authors, values: papers participates in.
    print(data["features"])
    print(data["labels"])
    # Instantiates the FormanRicci class
    forman_ricci = FormanRicci(data)
    # Computes the Forman-Ricci curvature
    forman_ricci.compute_forman_ricci()
    # Accesses the results
    print("Node Degrees:", forman_ricci.node_degrees)
    print("Forman-Ricci Curvature:", forman_ricci.forman_ricci)
    print("DONE")
