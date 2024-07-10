"""
encodings.py

This module contains functions for adding encodings
to a dataset (curvature, laplacians, random walks).
"""

import numpy as np
from curvatures import FormanRicci


class HypergraphCurvatureProfile:
    """
    This class computes the local curvature profile
    structural encoding for each node in a hypergraph.
    """
    def __init__(self):
        pass

    def compute_frc(self, hypergraph: dict) -> dict:
        """
        Compute the HCP based on the FRC.
        """
        frc = FormanRicci(hypergraph)
        frc.compute_forman_ricci()
        
        # for each node, get the hyperedges it belongs to
        num_nodes = hypergraph["n"]
        hyperedges = [hypergraph["hypergraph"] for _ in range(num_nodes) if _ in hypergraph["hypergraph"].values()]

        # for each node, get the min, max, mean, median,
        # and std of the FRC values of the hyperedges it belongs to
        frc_profile = {}
        for node in range(num_nodes):
            frc_values = [frc.forman_ricci[hyperedge] for hyperedge in hyperedges[node]]
            frc_profile[node] = [
                min(frc_values),
                max(frc_values),
                np.mean(frc_values),
                np.median(frc_values),
                np.std(frc_values)]

        # turn the FRC profile into a np.matrix and stack it with the features
        for node in range(num_nodes):
            frc_vals = np.matrix(frc_profile[node])
            hypergraph["features"][node] = np.hstack((hypergraph["features"][node], frc_vals))

        return hypergraph
    

    def compute_orc(self, hypergraph: dict) -> dict:
        """
        Compute the HCP based on the ORC.
        """
        pass