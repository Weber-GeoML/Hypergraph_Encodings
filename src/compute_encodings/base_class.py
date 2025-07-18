"""Base class for computing and saving encodings.

Processes one hypergraph at a time.
"""

import inspect
import multiprocessing as mp
import os
import pickle
import warnings

import numpy as np

from encodings_hnns.encodings import HypergraphEncodings
from encodings_hnns.laplacians import DisconnectedError

warnings.simplefilter("ignore")


class EncodingsSaverBase(object):
    """Base class for computing and saving encodings"""

    def __init__(self, data: str) -> None:
        """Initialises the data directory

        Arguments:
            data:
                coauthorship/cocitation
        """

        current = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        )
        current = os.path.dirname(os.path.dirname(current))
        # Makes the paths
        self.d: str = os.path.join(
            current, "data", data
        )  # Modify for where your data is!
        self.individual_files_dir = os.path.join(self.d, "individual_files")

        # Create individual_files directory if it doesn't exist
        os.makedirs(self.individual_files_dir, exist_ok=True)

        print(f"The data directory is {self.d}")
        print(f"The individual files directory is {self.individual_files_dir}")

        self.data = data

    ############################################################################
    ################### Process one hypergraph within file #####################
    ############################################################################

    def _process_hypergraph(
        self,
        hg,
        file_to_process: str,
        count: int,
        verbose: bool = False,
    ) -> tuple[
        list[dict],
        list[dict],
        list[dict],
        list[dict],
        list[dict],
        list[dict],
        list[dict],
        list[dict],
    ]:
        """Processes one hypergraph only.

        Used for multiprocessing.

        Args:
            hg:
                the hypergraph
            file_to_process:
                the name of the file
            count:
                the count.
            verbose:
                whether to print

        Returns:
            A tuple of lists of dictionaries, one for each encoding type.

            Each dictionary has the following fields:
                - hypergraph
                - features
                - labels
                - n

            The order of the lists is:
                - rw_EE
                - rw_EN
                - rw_WE
                - lape_hodge
                - lape_normalized
                - orc
                - frc
                - ldp

        """
        if verbose:
            print(f"The file is {file_to_process}")
            print(f"The count is {count}")
            print(f"Features shape: {hg['features'].shape}")
            print(f"Labels shape: {hg['labels'].shape}")

        # construct the hypergraph object in the same way we are used to
        hypergraph: dict = hg["hypergraph"]
        features: np.ndarray = hg["features"]
        # if isinstance(features, np.ndarray):
        #     if len(features.shape) == 1:
        #         features = features.reshape(-1, 1)
        # else:
        #     features = np.array(features).reshape(-1, 1)
        all_nodes: list = sorted(
            set(node for hyperedge in hypergraph.values() for node in hyperedge)
        )
        if verbose:
            print(f"we have {len(all_nodes)} nodes")
        features_shapes = features.shape
        if verbose:
            print(f"The features have shape {features.shape}")
        labels: np.ndarray = hg["labels"]
        # if len(labels.shape) == 1:
        #     labels = labels.reshape(-1, 1)
        dataset: dict = {
            "hypergraph": hypergraph,
            "features": features,
            "labels": labels,
            "n": features.shape[0],
        }

        # Add encodings for random walks, Laplacian, and curvature
        # they each contain only one element (one hg) at the end
        list_hgs_rw_EE, list_hgs_rw_EN, list_hgs_rw_WE = [], [], []
        list_hgs_lape_hodge, list_hgs_lape_normalized = [], []
        list_hgs_orc, list_hgs_frc = [], []
        list_hgs_ldp = []

        # add the encodings
        try:
            for random_walk_type in ["EE", "EN", "WE"]:
                hgencodings = HypergraphEncodings()
                k_rw = 20
                dataset_copy = dataset.copy()
                dataset_copy = hgencodings.add_randowm_walks_encodings(
                    dataset_copy,
                    rw_type=random_walk_type,
                    k=k_rw,
                    normalized=True,
                    dataset_name=None,
                    verbose=False,
                )
                features_shapes_with_encodings = dataset_copy["features"].shape
                assert features_shapes != features_shapes_with_encodings
                if random_walk_type == "WE":
                    list_hgs_rw_WE.append(dataset_copy)
                elif random_walk_type == "EE":
                    list_hgs_rw_EE.append(dataset_copy)
                elif random_walk_type == "EN":
                    list_hgs_rw_EN.append(dataset_copy)
                del hgencodings
        except DisconnectedError as e:
            print(f"Error: {e}")
            list_hgs_rw_WE.append(dataset_copy)
            list_hgs_rw_EE.append(dataset_copy)
            list_hgs_rw_EN.append(dataset_copy)

        try:
            for laplacian_type in ["Hodge", "Normalized"]:
                hgencodings = HypergraphEncodings()
                dataset_copy = dataset.copy()
                dataset_copy = hgencodings.add_laplacian_encodings(
                    dataset_copy,
                    laplacian_type=laplacian_type,
                    normalized=True,
                    dataset_name=None,
                    verbose=verbose,
                )
                if laplacian_type == "Hodge":
                    list_hgs_lape_hodge.append(dataset_copy)
                elif laplacian_type == "Normalized":
                    list_hgs_lape_normalized.append(dataset_copy)
                del hgencodings
        except DisconnectedError as e:
            print(f"Error: {e}")
            list_hgs_lape_hodge.append(dataset_copy)
            list_hgs_lape_normalized.append(dataset_copy)
        try:
            for curvature_type in ["ORC", "FRC"]:
                hgencodings = HypergraphEncodings()
                dataset_copy = dataset.copy()
                dataset_copy = hgencodings.add_curvature_encodings(
                    dataset_copy,
                    curvature_type=curvature_type,
                    normalized=True,
                    dataset_name=None,
                    verbose=verbose,
                )
                if curvature_type == "ORC":
                    list_hgs_orc.append(dataset_copy)
                elif curvature_type == "FRC":
                    list_hgs_frc.append(dataset_copy)
                del hgencodings
        except DisconnectedError as e:
            print(f"Error: {e}")
            list_hgs_orc.append(dataset_copy)
            list_hgs_frc.append(dataset_copy)
        try:
            hgencodings = HypergraphEncodings()
            dataset_copy = dataset.copy()
            dataset_copy = hgencodings.add_degree_encodings(
                dataset_copy,
                normalized=True,
                dataset_name=None,
                verbose=verbose,
            )
            list_hgs_ldp.append(dataset_copy)
        except DisconnectedError as e:
            print(f"Error: {e}")
            list_hgs_ldp.append(dataset_copy)

        encoding_map: dict = {
            "rw_EE": list_hgs_rw_EE,
            "rw_EN": list_hgs_rw_EN,
            "rw_WE": list_hgs_rw_WE,
            "lape_hodge": list_hgs_lape_hodge,
            "lape_normalized": list_hgs_lape_normalized,
            "orc": list_hgs_orc,
            "frc": list_hgs_frc,
            "ldp": list_hgs_ldp,
            # Add other mappings as needed
        }

        for encoding_type, encoding_list in encoding_map.items():
            save_file = (
                f"{file_to_process}_with_encodings_{encoding_type}_count_{count}.pickle"
            )
            with open(
                os.path.join(self.individual_files_dir, save_file), "wb"
            ) as handle:
                pickle.dump(encoding_list, handle)

        return (
            list_hgs_rw_EE,
            list_hgs_rw_EN,
            list_hgs_rw_WE,
            list_hgs_lape_hodge,
            list_hgs_lape_normalized,
            list_hgs_orc,
            list_hgs_frc,
            list_hgs_ldp,
        )

    ############################################################################
    ################### Process one file #######################################
    ############################################################################

    def _process_file(
        self, file_to_process: str
    ) -> tuple[list, list, list, list, list, list, list, list]:
        """Processes one file at a time.

        Used for parrallel computations.
        THe files are imdb, reddit and collab.

        Args:
            file_to_process:
                the file
        """
        list_hgs_rw_EE: list[dict] = []
        list_hgs_rw_EN: list[dict] = []
        list_hgs_rw_WE: list[dict] = []
        list_hgs_lape_hodge: list[dict] = []
        list_hgs_lape_normalized: list[dict] = []
        list_hgs_orc: list[dict] = []
        list_hgs_frc: list[dict] = []
        list_hgs_ldp: list[dict] = []

        with open(os.path.join(self.d, f"{file_to_process}.pickle"), "rb") as handle:
            # list of hypergraphs
            hypergraphs: list[dict] = pickle.load(handle)
            print(f"The file {file_to_process} contains {len(hypergraphs)} hypergraphs")
            results = []
            with mp.Pool() as pool:
                for count, hg in enumerate(hypergraphs):
                    result = pool.apply_async(
                        self._process_hypergraph,
                        (hg, file_to_process, count),
                    )
                    results.append(result)
                pool.close()
                pool.join()

        # Retrieve and accumulate results
        for result in results:
            (
                rw_EE,
                rw_EN,
                rw_WE,
                lape_hodge,
                lape_normalized,
                orc,
                frc,
                ldp,
            ) = result.get()

            list_hgs_rw_EE.extend(rw_EE)
            list_hgs_rw_EN.extend(rw_EN)
            list_hgs_rw_WE.extend(rw_WE)
            list_hgs_lape_hodge.extend(lape_hodge)
            list_hgs_lape_normalized.extend(lape_normalized)
            list_hgs_orc.extend(orc)
            list_hgs_frc.extend(frc)
            list_hgs_ldp.extend(ldp)

        # Save each accumulated encoding list as a separate pickle file
        accumulated_encodings = {
            "rw_EE": list_hgs_rw_EE,
            "rw_EN": list_hgs_rw_EN,
            "rw_WE": list_hgs_rw_WE,
            "lape_hodge": list_hgs_lape_hodge,
            "lape_normalized": list_hgs_lape_normalized,
            "orc": list_hgs_orc,
            "frc": list_hgs_frc,
            "ldp": list_hgs_ldp,
        }

        for encoding_type, encoding_list in accumulated_encodings.items():
            combined_file = f"{file_to_process}_with_encodings_{encoding_type}.pickle"
            with open(os.path.join(self.d, combined_file), "wb") as handle:
                pickle.dump(encoding_list, handle)
                print(f"Saved combined encodings to {combined_file}")
                print(
                    f"Saved combined encodings to {combined_file} with {len(encoding_list)} elements"
                )

        return (
            list_hgs_rw_EE,
            list_hgs_rw_EN,
            list_hgs_rw_WE,
            list_hgs_lape_hodge,
            list_hgs_lape_normalized,
            list_hgs_orc,
            list_hgs_frc,
            list_hgs_ldp,
        )
