"""
encodings.py

This module contains functions for adding encodings
to a dataset (curvature, laplacians, random walks).
"""

# NOTE: the following is actually not True! We
# need to investigate what is the meaning of the other features...
# assert that all_nodes is {0,1,2,3..., number of nodes}

import json
import os
import pickle
import random

import numpy as np

from encodings_hnns.curvatures_frc import FormanRicci
from encodings_hnns.curvatures_orc import ORC
from encodings_hnns.laplacians import Laplacians


class HypergraphEncodings:
    """Computes the local curvature profile

    Computes the structural encoding for each node in a hypergraph.
    """

    def __init__(self):
        self.hyperedges: None | dict = None

    def compute_hyperedges(self, hypergraph: dict, verbose: bool = True) -> None:
        """Computes a dictionary called hyperedges.

        The dictionary contains as keys the nodes,
        as values the hyperedges the node belongs to.

        Args:
            hypergraph:
                the hypergraph object
            verbose:
                print more

        Sets:
            the dict that contains nodes and their hyperedges

        We do not want to compute this twice so if we compute
        it for eg FRC, we can reuse it for ORC.
        """
        if verbose:
            print(f"The hypergraph has {len(hypergraph['hypergraph'])} (hyper)edges")

        # for each node, get the hyperedges it belongs to
        # keys are node, values are hyperedges name the node belongs to
        hyperedges: dict = {}

        # adding this to debug
        all_nodes: list = sorted(
            set(
                node
                for hyperedge in hypergraph["hypergraph"].values()
                for node in hyperedge
            )
        )

        # loops through hyperedges
        for hyperedge_name, hyperedge in hypergraph["hypergraph"].items():
            if verbose:
                print(f"The hyperedge is {hyperedge}")
            # loops through nodes in the hyperedge
            for node in hyperedge:
                if verbose:
                    print(f"node is {node}")
                assert isinstance(
                    node, (int, np.int32)
                ), f"Node {node} is not an integer, it is of type {type(node)}"
                if node not in hyperedges:
                    hyperedges[node] = []
                hyperedges[node].append(hyperedge_name)
        self.hyperedges = hyperedges
        assert len(self.hyperedges) == len(
            all_nodes
        ), f"We have {len(self.hyperedges)} vs {len(all_nodes)}"

    def add_degree_encodings(
        self,
        hypergraph: dict,
        verbose: bool = False,
        normalized: bool = True,
        dataset_name: str | None = None,
    ) -> dict:
        """Computes the LDP. This is the degree profile.

        Args:
            hypergraph:
                hypergraph dict containing hypergraph, features, label, n
            verbose:
                to print more
            normalized:
                technical detail about needeing [] when we normalize in some cases
            dataset_name:
                the name of the dataset. Used for savings the encodings
        Returns:
            the hypergraph with the degree profile encodings added to the featuress

        I am adding tht ability to save the encodings. Ie, we only compute them once.
        dataset["features"]

        """
        filename: str = (
            f"computed_encodings/{dataset_name}_degree_encodings_normalized_{normalized}.pkl"
        )
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                print(f"Loading hypergraph from {filename}")
                return pickle.load(f)

        else:
            # compute the encodings and save
            assert (
                hypergraph["features"].shape[0] == hypergraph["n"]
            ), f"BEFORE: The shape is {hypergraph['features'].shape[0]} but n is {hypergraph['n']}"

            laplacian: Laplacians = Laplacians(hypergraph)
            laplacian.compute_ldp()

            # for each node, get the min, max, mean, median,
            # and std of the degrees of the neighbors
            ld_profile: dict = laplacian.ldp

            if self.hyperedges == None:
                self.compute_hyperedges(hypergraph)

            features_augmented = hypergraph["features"]
            # Determines the target shape
            target_shape = (
                features_augmented.shape[0],
                features_augmented.shape[1] + 6,
            )

            # Creates a new array of zeros with the target shape
            padded_features = np.zeros(target_shape)

            # Copies the original features into the new padded array
            padded_features[:, : features_augmented.shape[1]] = features_augmented

            # turn the degree profile into a np.matrix and stack it with the features
            # loops through node
            if len(hypergraph["features"]) == 0:
                print("Will be implemented")
                raise NotImplementedError
            for node in self.hyperedges.keys():
                ld_vals = np.matrix(ld_profile[node])
                if verbose:
                    print(
                        f"The hypergraph features for node {node}, are \n {hypergraph['features'][node]}"
                    )
                    print(f"We add the degree encoding:\n {ld_vals}")
                if normalized:
                    stacked_features = np.hstack(
                        (hypergraph["features"][node], ld_vals)
                    )
                elif not normalized:
                    stacked_features = np.hstack(
                        ([hypergraph["features"][node]], ld_vals)
                    )
                if verbose:
                    print(f"The stacked features are \n {stacked_features}")
                padded_features[node] = stacked_features
            hypergraph["features"] = padded_features
            assert (
                hypergraph["features"].shape[0] == hypergraph["n"]
            ), f"The shape is {hypergraph['features'].shape[0]} but n is {hypergraph['n']}"

            if dataset_name is not None:
                with open(filename, "wb") as f:
                    pickle.dump(hypergraph, f)
                print(f"Hypergraph saved as {filename}")
            return hypergraph

    def add_curvature_encodings(
        self,
        hypergraph: dict,
        verbose: bool = True,
        type: str = "FRC",
        normalized: bool = True,
        dataset_name: str | None = None,
    ) -> dict:
        """Computes the LCP based on the FRC or ORC.

        Args:
            hypergraph:
                hypergraph dict containing hypergraph, features, label, n
            verbose:
                to print more
            type:
                the type of encoding we add.
                Currently can be ORC or FRC
            normalized:
                when false, need to slight\ly modify the code
            dataset_name:
                the name of the dataset. Used for savings the encodings

        Returns:
            the hypergraph with the frc or orc encodings added to the featuress

        """
        filename: str = (
            f"computed_encodings/{dataset_name}_curvature_encodings_{type}_normalized_{normalized}.pkl"
        )
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                print(f"Loading hypergraph from {filename}")
                return pickle.load(f)
        else:
            assert (
                hypergraph["features"].shape[0] == hypergraph["n"]
            ), f"BEFORE: The shape is {hypergraph['features'].shape[0]} but n is {hypergraph['n']}"

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
                    print("the hyperedges are")
                    print(self.hyperedges)

            if verbose:
                print(f"the hyperedges are {self.hyperedges}")

            # for each node, get the min, max, mean, median,
            # and std of the FRC or ORC values of the hyperedges it belongs to
            rc_profile: dict[list[float]] = {}
            for node in self.hyperedges.keys():
                if type == "FRC":
                    rc_values = [
                        rc.forman_ricci[hyperedge]
                        for hyperedge in self.hyperedges[node]
                    ]
                elif type == "ORC":
                    rc_values = [
                        rc.edge_curvature[hyperedge]
                        for hyperedge in self.hyperedges[node]
                    ]
                rc_profile[node] = [
                    min(rc_values),
                    max(rc_values),
                    np.mean(rc_values),
                    np.median(rc_values),
                    np.std(rc_values),
                ]

            features_augmented = hypergraph["features"]
            # Determines the target shape
            target_shape = (
                features_augmented.shape[0],
                features_augmented.shape[1] + 5,
            )

            # Creates a new array of zeros with the target shape
            padded_features = np.zeros(target_shape)

            # Copies the original features into the new padded array
            padded_features[:, : features_augmented.shape[1]] = features_augmented

            # turn the RC profile into a np.matrix and stack it with the features
            if len(hypergraph["features"]) == 0:
                print("Will be implemented")
                raise NotImplementedError
            for node in self.hyperedges.keys():
                rc_vals = np.matrix(rc_profile[node])
                if verbose:
                    print(
                        f"The hypergraph features for node {node}, are \n {hypergraph['features'][node]}"
                    )
                    print(f"We add the encoding:\n {rc_vals}")
                if normalized:
                    padded_features[node] = np.hstack(
                        (hypergraph["features"][node], rc_vals)
                    )
                elif not normalized:
                    padded_features[node] = np.hstack(
                        ([hypergraph["features"][node]], rc_vals)
                    )
            hypergraph["features"] = padded_features
            assert (
                hypergraph["features"].shape[0] == hypergraph["n"]
            ), f"The shape is {hypergraph['features'].shape[0]} but n is {hypergraph['n']}"
            if dataset_name is not None:
                with open(filename, "wb") as f:
                    pickle.dump(hypergraph, f)
                print(f"Hypergraph saved as {filename}")
            return hypergraph

    def add_laplacian_encodings(
        self,
        hypergraph: dict,
        verbose: bool = True,
        type: str = "Hodge",
        rw_type: str = "EN",
        normalized: bool = True,
        dataset_name: str | None = None,
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
                the type of random walk on the hypergraph
                Only use if type is "RW"
                EN (Equal Node), WE (Weighted Edge) , EE (Equal Edge)
            normalized:
                when false, need to slight\ly modify the code
            dataset_name:
                the name of the dataset. Used for savings the encodings

        Returns:
            the hypergraph with the Laplacian encodings added to the featuress
        """
        if type == "Hodge" or type == "Normalized":
            filename: str = (
                f"computed_encodings/{dataset_name}_laplacian_encodings_{type}_normalized_{normalized}.pkl"
            )
        elif type == "RW":
            filename: str = (
                f"computed_encodings/{dataset_name}_laplacian_encodings_{type}_rw_{rw_type}_normalized_{normalized}.pkl"
            )
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                print(f"Loading hypergraph from {filename}")
                return pickle.load(f)

        else:
            assert (
                hypergraph["features"].shape[0] == hypergraph["n"]
            ), f"BEFORE: The shape is {hypergraph['features'].shape[0]} but n is {hypergraph['n']}"

            # Computes the dictionary with keys as node
            # and values as hyperedges
            if self.hyperedges == None:
                self.compute_hyperedges(hypergraph)

            laplacian: Laplacians = Laplacians(hypergraph=hypergraph)
            eigenvalues: np.ndarray
            eigenvectors: np.ndarray
            if type == "Hodge":
                laplacian.compute_hodge_laplacian()
                # We would use up for edge feature
                # as the up matrix is number of edge by number of egde
                # We use down for node feature
                # as the down matrix is number of nodes by number of nodes
                if verbose:
                    print(
                        f"The Hodge Laplacian (down) is \n {laplacian.hodge_laplacian_down}"
                    )
                # Compute the eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eig(
                    laplacian.hodge_laplacian_down
                )
            elif type == "Normalized":
                laplacian.compute_normalized_laplacian()
                print(f"The normalized Laplacian is {laplacian.normalized_laplacian}")
                eigenvalues, eigenvectors = np.linalg.eig(
                    laplacian.normalized_laplacian
                )
            elif type == "RW":
                laplacian.compute_random_walk_laplacian(type=rw_type)
                if verbose:
                    print(f"The RW laplacian is \n {laplacian.rw_laplacian}")
                    print(f"The RW laplacian is \n {laplacian.rw_laplacian})")

            # TODO: take the real part of the eigenvalues/eigenvectors
            # put a flag to catch if it larger than 10e-3 (imaginary part)

            # Print the results
            if verbose:
                print("Eigenvalues:")
                print(eigenvalues)
                print("Eigenvectors:")
                print(eigenvectors)

            # That was true for Hodge
            # if type == "Normalized" or type == "Hodge":
            #     # Creates a diagonal matrix from the eigenvalues
            #     diagonal_matrix = np.diag(eigenvalues)

            #     # Reconstructs the original matrix
            #     reconstructed_matrix = eigenvectors @ diagonal_matrix @ eigenvectors.T

            #     # Compare reconstructed matrix to the original matrix
            #     if np.allclose(reconstructed_matrix, laplacian.hodge_laplacian_down):
            #         print("Reconstructed matrix is close to the original matrix.")
            #         print("Symmetric matrix. Expected for Hodge, normalized")
            #     else:
            #         print("Reconstructed matrix differs from the original matrix.")

            # We randonly flip the sign of the eigenvectors
            # this means that if we use k eigenvectors, we have
            # 2^k different possibilities
            sign: int = random.choice([-1, 1])

            features_augmented = hypergraph["features"]
            # Determines the target shape
            target_shape = (
                features_augmented.shape[0],
                features_augmented.shape[1] + eigenvectors[0].shape[0],
            )

            # Creates a new array of zeros with the target shape
            padded_features = np.zeros(target_shape)

            # Copies the original features into the new padded array
            padded_features[:, : features_augmented.shape[1]] = features_augmented

            # turn the FRC profile into a np.matrix and stack it with the features
            if len(hypergraph["features"]) == 0:
                print("Will be implemented")
                raise NotImplementedError
            i = 0
            for node in self.hyperedges.keys():
                laplacian_vals = eigenvectors[:, i].reshape(1, -1)
                laplacian_vals = sign * laplacian_vals
                if verbose:
                    print(
                        f"The hypergraph features for node {node} are \n {hypergraph['features'][node]}"
                    )
                    print(f"We add the Laplacian based encoding:\n {laplacian_vals}")
                # this assumes that the features are present
                if normalized:
                    stacked_features = np.hstack(
                        (hypergraph["features"][node], laplacian_vals)
                    )
                elif not normalized:
                    stacked_features = np.hstack(
                        ([hypergraph["features"][node]], laplacian_vals)
                    )
                if verbose:
                    print(f"The stacked features are {stacked_features}")
                padded_features[node] = stacked_features
                i += 1

            hypergraph["features"] = padded_features
            assert (
                hypergraph["features"].shape[0] == hypergraph["n"]
            ), f"The shape is {hypergraph['features'].shape[0]} but n is {hypergraph['n']}"
            if dataset_name is not None:
                with open(filename, "wb") as f:
                    pickle.dump(hypergraph, f)
                print(f"Hypergraph saved as {filename}")
            return hypergraph

    def add_randowm_walks_encodings(
        self,
        hypergraph: dict,
        verbose: bool = True,
        rw_type: str = "WE",
        k: int = 20,
        normalized: bool = True,
        dataset_name: str | None = None,
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
                EN
            k:
                number of steps of random walk
            normalized:
                when false, need to slight\ly modify the code
            dataset_name:
                the name of the dataset. Used for savings the encodings

        Returns:
            the hypergraph with the RW encodings added to the featuress

        """
        filename: str = (
            f"computed_encodings/{dataset_name}_rw_encodings_{rw_type}_k_{k}_normalized_{normalized}.pkl"
        )
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                print(f"Loading hypergraph from {filename}")
                return pickle.load(f)

        else:
            assert (
                hypergraph["features"].shape[0] == hypergraph["n"]
            ), f"BEFORE: The shape is {hypergraph['features'].shape[0]} but n is {hypergraph['n']}"

            if self.hyperedges == None:
                self.compute_hyperedges(hypergraph)

            laplacian: Laplacians = Laplacians(hypergraph=hypergraph)
            # get the laplacian. Take the opposite and add I:
            # gives probability of going from i to j
            if verbose:
                print(f"We are doing a {rw_type} rw")
            laplacian.compute_random_walk_laplacian(type=rw_type)
            num_nodes: int = len(self.hyperedges.keys())
            all_nodes: list = sorted(
                set(
                    node
                    for hyperedge in hypergraph["hypergraph"].values()
                    for node in hyperedge
                )
            )
            assert (
                len(all_nodes) == num_nodes
            ), f"We have {len(all_nodes)} vs {num_nodes}"
            try:
                rw_matrix: np.ndarray = -laplacian.rw_laplacian + np.eye(num_nodes)
            except:
                print(self.hyperedges.keys())
                assert False

            if verbose:
                print(f"The random walk matrix is \n {rw_matrix}")
                print(rw_matrix)

            matrix_powers: list = []

            for hop in range(k):
                rw_matrix_k = np.linalg.matrix_power(rw_matrix, hop)
                matrix_powers.append(np.diag(rw_matrix_k))

            # Converts the list of matrix powers to a numpy array
            matrix_powers = np.array(matrix_powers)

            assert matrix_powers.shape[0] == k

            features_augmented = hypergraph["features"]
            # Determines the target shape
            target_shape = (
                features_augmented.shape[0],
                features_augmented.shape[1] + k,
            )

            # Creates a new array of zeros with the target shape
            padded_features = np.zeros(target_shape)

            # Copies the original features into the new padded array
            padded_features[:, : features_augmented.shape[1]] = features_augmented

            if len(hypergraph["features"]) == 0:
                print("Will be implemented")
                raise NotImplementedError

            i: int = 0
            for node in self.hyperedges.keys():
                if verbose:
                    print(f"The node is {node}")
                assert matrix_powers[:, i].shape == (k,)
                laplacian_vals = matrix_powers[:, i].reshape(1, -1)
                if verbose:
                    print(
                        f"The hypergraph features for node {node} are \n {hypergraph['features'][node]}"
                    )
                    print(f"The shape is {len(hypergraph['features'][node])}")
                    print(
                        f"We add the RW based encoding:\n {laplacian_vals} \n with shape {laplacian_vals.shape}"
                    )
                if normalized:
                    stacked_features = np.hstack(
                        (hypergraph["features"][node], laplacian_vals)
                    )
                elif not normalized:
                    stacked_features = np.hstack(
                        ([hypergraph["features"][node]], laplacian_vals)
                    )
                padded_features[node] = stacked_features
                i += 1

            hypergraph["features"] = padded_features
            assert (
                hypergraph["features"].shape[0] == hypergraph["n"]
            ), f"The shape is {hypergraph['features'].shape[0]} but n is {hypergraph['n']}"
            if dataset_name is not None:
                with open(filename, "wb") as f:
                    pickle.dump(hypergraph, f)
                print(f"Hypergraph saved as {filename}")
            return hypergraph


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

    hg: dict[str, dict | int] = {
        "hypergraph": {
            "yellow": [0, 1, 2, 3],
            "red": [1, 2],
            "green": [2, 4, 5],
            "blue": [3, 4],
        },
        "features": np.matrix([[1], [1], [1], [1], [1], [1]]),
        "labels": {},
        "n": 6,
    }
    # print(hg["features"])
    # print(len(hg["features"]))
    # assert False
    # Instantiates the Hypergraph Curvature Profile class
    hgcurvaturprofile = HypergraphEncodings()
    hg = hgcurvaturprofile.add_randowm_walks_encodings(hg)
    assert hg["features"].shape[0] == hg["n"]
    assert hg["features"].shape[1] == 21, f"The shape is {hg['features'].shape[1]}"
    hg = hgcurvaturprofile.add_degree_encodings(hg)
    assert hg["features"].shape[0] == hg["n"]
    assert hg["features"].shape[1] == 27, f"The shape is {hg['features'].shape[1]}"

    hg = hgcurvaturprofile.add_laplacian_encodings(hg)
    assert hg["features"].shape[0] == hg["n"]
    assert hg["features"].shape[1] == 27, f"The shape is {hg['features'].shape[1]}"

    hg = hgcurvaturprofile.add_curvature_encodings(hg)
    assert hg["features"].shape[0] == hg["n"]
    assert hg["features"].shape[1] == 27, f"The shape is {hg['features'].shape[1]}"

    # hg = hgcurvaturprofile.add_laplacian_encodings(hg)
    hg = hgcurvaturprofile.add_randowm_walks_encodings(hg)
    assert hg["features"].shape[0] == hg["n"]
    print(hg)
