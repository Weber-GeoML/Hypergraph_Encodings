"""
curvatures.py

This module contains functions for computing the Forman-Ricci and
Ollivier-Ricci curvatures of an undirected hypergraph.
"""

import numpy as np

class FormanRicci:
    def __init__(self, hypergraph):
        """
        Initialize the Forman-Ricci curvature object.
        A hypergraph is a dictionary with the following keys:
        - hypergraph : a dictionary with the hyperedges as values
        - features : a dictionary with the features of the nodes as values
        - labels : a dictionary with the labels of the nodes as values
        - n : the number of nodes in the hypergraph
        """
        assert 'hypergraph' in hypergraph.keys(), "Hypergraph not found."
        assert 'features' in hypergraph.keys(), "Features not found."
        assert 'labels' in hypergraph.keys(), "Labels not found."
        assert 'n' in hypergraph.keys(), "Number of nodes not found."

        self.hypergraph = hypergraph
        self.node_degrees = {}
        self.forman_ricci = {}

    def compute_forman_ricci(self):
        """
        Compute the Forman-Ricci curvature of a hyperedge e according to the formula
        F(e) = 2|e| - D
        where |e| is the number of nodes in the hyperedge and D is the sum of the degrees.
        """
        assert self.forman_ricci == {}, "Forman-Ricci curvature already computed."

        if self.node_degrees == {}:
            self.compute_node_degrees()
        
        for hyperedge in self.hypergraph['hypergraph'].values():
            hyperedge_degree = len(hyperedge)
            hyperedge_sum_degrees = sum([self.node_degrees[node] for node in hyperedge])
            self.forman_ricci[hyperedge] = 2 * hyperedge_degree - hyperedge_sum_degrees

    def compute_node_degrees(self):
        """
        Compute the degree of each node in the hypergraph.
        """
        assert self.node_degrees == {}, "Node degrees already computed."

        for hyperedge in self.hypergraph['hypergraph'].values():
            for node in hyperedge:
                if node not in self.node_degrees:
                    self.node_degrees[node] = 1
                self.node_degrees[node] += 1