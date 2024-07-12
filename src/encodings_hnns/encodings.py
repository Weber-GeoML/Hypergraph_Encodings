"""
encodings.py

This module contains functions for adding encodings
to a dataset (curvature, laplacians, random walks).
"""

import numpy as np
from encodings_hnns.curvatures import FormanRicci


class HypergraphCurvatureProfile:
    """
    This class computes the local curvature profile
    structural encoding for each node in a hypergraph.
    """

    def __init__(self):
        pass

    def compute_frc(self, hypergraph: dict, verbose: bool = True) -> dict:
        """
        Compute the HCP based on the FRC.

        Args:
            hypergraph:
                hypergraph dict containing hypergraph, features, label, n
            verbose:
                to print more

        """
        frc = FormanRicci(hypergraph)
        frc.compute_forman_ricci()

        # for each node, get the hyperedges it belongs to
        num_nodes: int = hypergraph["n"]
        # keys are node, values are hyperedges name the node belongs to
        hyperedges: dict = {}

        # loops through hyperedges
        for hyperedge_name, hyperedge in hypergraph["hypergraph"].items():
            # loops through nodes in the hyperedge
            for node in hyperedge:
                if node not in hyperedges:
                    hyperedges[node] = []
                hyperedges[node].append(hyperedge_name)

        if verbose:
            print(f"the hyperedges are {hyperedges}")

        # for each node, get the min, max, mean, median,
        # and std of the FRC values of the hyperedges it belongs to
        frc_profile: dict[list[float]] = {}
        for node in hyperedges.keys():
            frc_values = [frc.forman_ricci[hyperedge] for hyperedge in hyperedges[node]]
            frc_profile[node] = [
                min(frc_values),
                max(frc_values),
                np.mean(frc_values),
                np.median(frc_values),
                np.std(frc_values),
            ]

        # turn the FRC profile into a np.matrix and stack it with the features
        for node in hyperedges.keys():
            frc_vals = np.matrix(frc_profile[node])
            print(hypergraph["features"][node])
            print(frc_vals)
            hypergraph["features"][node] = np.hstack(
                (hypergraph["features"][node], frc_vals)
            )

        return hypergraph

    def compute_orc(self, hypergraph: dict) -> dict:
        """
        Compute the HCP based on the ORC.
        """
        pass


# Example utilization
if __name__ == "__main__":

    hg: dict[str, dict | int] = {
        "hypergraph": {
            "yellow": [1, 2, 3],
            "red": [2, 3],
            "green": [3, 5, 6],
            "blue": [4, 5],
        },
        "features": {1: [[1]], 2: [[1]], 3: [[1]], 4: [[1]], 5: [[1]], 6: [[1]]},
        "labels": {},
        "n": 6,
    }

    # Instantiates the Hypergraph Curvature Profile class
    hgcurvaturprofile = HypergraphCurvatureProfile()
    hg = hgcurvaturprofile.compute_frc(hg)
    print(hg)
