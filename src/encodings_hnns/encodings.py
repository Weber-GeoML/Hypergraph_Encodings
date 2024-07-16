"""
encodings.py

This module contains functions for adding encodings
to a dataset (curvature, laplacians, random walks).
"""

import numpy as np

from encodings_hnns.curvatures import FormanRicci
from encodings_hnns.curvatures_orc import ORC
from encodings_hnns.laplacians import Laplacians


class HypergraphCurvatureProfile:
    """Computes the local curvature profile

    Computes the structural encoding for each node in a hypergraph.
    """

    def __init__(self):
        self.hyperedges: None | dict = None

    def compute_hyperedges(self, hypergraph: dict) -> None:
        """Computes a dictionary called hyperedges

        The dictionary contains as keys the nodes,
        as values the hyperedges the node belongs to.

        Args:
            hypergraph:
                the hypergraph object

        Sets:
            the dict that contains nodes and their hyperedges

        We do not want to compute this twice so if we compute
        it for FRC, we can reuse it for ORC.
        """
        # for each node, get the hyperedges it belongs to
        # keys are node, values are hyperedges name the node belongs to
        hyperedges: dict = {}

        # loops through hyperedges
        for hyperedge_name, hyperedge in hypergraph["hypergraph"].items():
            # loops through nodes in the hyperedge
            for node in hyperedge:
                if node not in hyperedges:
                    hyperedges[node] = []
                hyperedges[node].append(hyperedge_name)

        self.hyperedges = hyperedges

    def add_curvature_encodings(
        self, hypergraph: dict, verbose: bool = True, type: str = "FRC"
    ) -> dict:
        """Computes the HCP based on the FRC.

        Args:
            hypergraph:
                hypergraph dict containing hypergraph, features, label, n
            verbose:
                to print more
            type:
                the type of encoding we add.
                Currently can be ORC or FRC

        Returns:
            the hypergraph with the frc encodings adding to the featuress

        """
        rc: FormanRicci | ORC
        if type == "FRC":
            rc = FormanRicci(hypergraph)
            rc.compute_forman_ricci()
        elif type == "ORC":
            rc = ORC(hypergraph)
            # following calls the julia code and runs the subroutines
            rc.compute_orc()

        if self.hyperedges == None:
            self.compute_hyperedges(hypergraph)

        if verbose:
            print(f"the hyperedges are {self.hyperedges}")

        # for each node, get the min, max, mean, median,
        # and std of the FRC values of the hyperedges it belongs to
        rc_profile: dict[list[float]] = {}
        for node in self.hyperedges.keys():
            if type == "FRC":
                rc_values = [
                    rc.forman_ricci[hyperedge] for hyperedge in self.hyperedges[node]
                ]
            elif type == "ORC":
                rc_values = [
                    rc.edge_curvature[hyperedge] for hyperedge in self.hyperedges[node]
                ]
            rc_profile[node] = [
                min(rc_values),
                max(rc_values),
                np.mean(rc_values),
                np.median(rc_values),
                np.std(rc_values),
            ]

        # turn the FRC profile into a np.matrix and stack it with the features
        for node in self.hyperedges.keys():
            rc_vals = np.matrix(rc_profile[node])
            if verbose:
                print(
                    f"The hypergraph features for node {node} are \n {hypergraph['features'][node]}"
                )
                print(f"We add the encoding:\n {rc_vals}")
            hypergraph["features"][node] = np.hstack(
                (hypergraph["features"][node], rc_vals)
            )

        return hypergraph

    def add_laplacian_encodings(
        self,
        hypergraph: dict,
        verbose: bool = True,
        type: str = "Normalized",
        rw_type: str | None = None,
        alpha: float | None = None,
    ) -> dict:
        """Adds encodings based on Laplacians

        Args:
            hypergraph:
                hypergraph dict containing hypergraph, features, label, n
            verbose:
                to print more
            type:
                the type of Laplacian we use for the encodings
                Hodge, RW, Normalized
            rw_type:
                EN, WE, EE
            alpha:
                for the random walk

        """
        if self.hyperedges == None:
            self.compute_hyperedges(hypergraph)

        laplacian: Laplacians = Laplacians(hypergraph=hypergraph)
        if type == "Hodge":
            laplacian.compute_hodge_laplacian()
            # We would use up for edge feature
            # as the up matrix is number of edge by number of egde
            print(laplacian.hodge_laplacian_up)
            # We use down for node feature
            # as the down matrix is number of nodes by number of nodes
            print(laplacian.hodge_laplacian_down)
            # Compute the eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(laplacian.hodge_laplacian_down)
        elif type == "Normalized":
            # TODO
            laplacian.compute_normalized_laplacian()
            eigenvalues, eigenvectors = np.linalg.eig(laplacian.normalized_laplacian)
        elif type == "RW":
            # Not symmetric. Need to think what we do
            pass
            # TODO

        # Print the results
        print("Eigenvalues:")
        print(eigenvalues)
        print("Eigenvectors:")
        print(eigenvectors)

        # Creates a diagonal matrix from the eigenvalues
        diagonal_matrix = np.diag(eigenvalues)

        # Reconstructs the original matrix
        reconstructed_matrix = eigenvectors @ diagonal_matrix @ eigenvectors.T

        # Compare reconstructed matrix to the original matrix
        if np.allclose(reconstructed_matrix, laplacian.hodge_laplacian_down):
            print("Reconstructed matrix is close to the original matrix.")
            print("Symmetric matrix. Expected for Hodge, normalized")
        else:
            print("Reconstructed matrix differs from the original matrix.")

        # turn the FRC profile into a np.matrix and stack it with the features
        i = 0
        for node in self.hyperedges.keys():
            laplacian_vals = eigenvectors[:, i].reshape(1, -1)
            if verbose:
                print(
                    f"The hypergraph features for node {node} are \n {hypergraph['features'][node]}"
                )
                print(f"We add the Laplacian based encoding:\n {laplacian_vals}")
            hypergraph["features"][node] = np.hstack(
                (hypergraph["features"][node], laplacian_vals)
            )
            i += 1
        return hypergraph

    def add_randowm_walks_encodings(
        self,
        hypergraph: dict,
        verbose: bool = True,
        rw_type: str = "WE",
        alpha: float = 0,
    ) -> dict:
        """Adds encodings based on RW

        Args:
            hypergraph:
                hypergraph dict containing hypergraph, features, label, n
            verbose:
                to print more
            rw_type:
                WE
                EE
            alpha:
                for the lazy RW or not.

        """
        pass
        # TODO


# Example utilization
if __name__ == "__main__":

    hg: dict[str, dict | int] = {
        "hypergraph": {
            "yellow": [1, 2, 3, 5],
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
    # hg = hgcurvaturprofile.compute_orc(hg)

    hg = hgcurvaturprofile.add_laplacian_encodings(hg)
    print(hg)
