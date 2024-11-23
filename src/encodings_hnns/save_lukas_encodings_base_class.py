import inspect
import multiprocessing as mp
import os
import pickle
import warnings

import numpy as np

from encodings_hnns.encodings import HypergraphEncodings
from encodings_hnns.laplacians import DisconnectedError

warnings.simplefilter("ignore")

# Define mapping of encoding types to their keys
ENCODING_MAP: dict[str, dict[str, str]] = {
    "Laplacian": {"Hodge": "lape_hodge", "Normalized": "lape_normalized"},
    "LCP": {"ORC": "orc", "FRC": "frc"},
    "LDP": {"LDP": "ldp"},
    "RW": {"EE": "rw_EE", "EN": "rw_EN", "WE": "rw_WE"},
}


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
        self.d: str = os.path.join(current, "data", data)
        self.individual_files_dir = os.path.join(self.d, "individual_files")

        # Create individual_files directory if it doesn't exist
        os.makedirs(self.individual_files_dir, exist_ok=True)

        self.data = data

    def compute_random_walks(
        self, dataset: dict, verbose: bool = False, random_walk_type: str = "WE"
    ) -> list:
        list_hgs_with_encodings: list = []
        try:
            hgencodings = HypergraphEncodings()
            k_rw = 20
            dataset_copy = dataset.copy()
            if verbose:
                print(f"Computing {random_walk_type} random walk encodings...")
                print(f"Input features shape: {dataset_copy['features'].shape}")

            features_shapes = dataset_copy["features"].shape

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
            list_hgs_with_encodings.append(dataset_copy)
            del hgencodings
        except DisconnectedError as e:
            print(f"Error: {e}")

        return list_hgs_with_encodings

    def compute_laplacian(
        self, dataset: dict, verbose: bool = False, laplacian_type: str = "Hodge"
    ) -> list:
        list_hgs_with_encodings: list = []
        try:
            hgencodings = HypergraphEncodings()
            dataset_copy = dataset.copy()
            dataset_copy = hgencodings.add_laplacian_encodings(
                dataset_copy,
                type=laplacian_type,
                normalized=True,
                dataset_name=None,
                verbose=verbose,
            )
            list_hgs_with_encodings.append(dataset_copy)
            del hgencodings
        except DisconnectedError as e:
            print(f"Error: {e}")
        return list_hgs_with_encodings

    def compute_lcp(
        self, dataset: dict, verbose: bool = False, curvature_type: str = "FRC"
    ) -> list:
        list_hgs_with_encodings: list = []
        try:
            hgencodings = HypergraphEncodings()
            dataset_copy = dataset.copy()
            dataset_copy = hgencodings.add_curvature_encodings(
                dataset_copy,
                type=curvature_type,
                normalized=True,
                dataset_name=None,
                verbose=verbose,
            )
            list_hgs_with_encodings.append(dataset_copy)
        except DisconnectedError as e:
            print(f"Error: {e}")
        return list_hgs_with_encodings

    def compute_ldp(self, dataset: dict, verbose: bool = False) -> list:
        list_hgs_with_encodings: list = []
        try:
            hgencodings = HypergraphEncodings()
            dataset_copy = dataset.copy()
            dataset_copy = hgencodings.add_degree_encodings(
                dataset_copy,
                normalized=True,
                dataset_name=None,
                verbose=verbose,
            )
            list_hgs_with_encodings.append(dataset_copy)
        except DisconnectedError as e:
            print(f"Error: {e}")
        return list_hgs_with_encodings

    def _process_hypergraph(
        self,
        hg,
        lukas_file: str,
        count: int,
        verbose: bool = False,
        encodings_to_compute: str = "LDP",
        laplacian_type: str = "Hodge",
        random_walk_type: str = "WE",
        curvature_type: str = "FRC",
    ) -> list:
        """Processes one hypergraph only.

        Used for multiprocessing.

        Args:
            hg:
                the hypergraph
            lukas_file:
                the name of the file
            count:
                the count.
            verbose:
                whether to print

        """
        if verbose:
            print(f"The file is {lukas_file}")
            print(f"The count is {count}")
            print(f"Features shape: {hg['features'].shape}")
            print(f"Labels shape: {hg['labels'].shape}")

        # construct the hypergraph object in the same way we are used to
        hypergraph: dict = hg["hypergraph"]
        features: np.ndarray = hg["features"]
        if isinstance(features, np.ndarray):
            if len(features.shape) == 1:
                features = features.reshape(-1, 1)
        else:
            features = np.array(features).reshape(-1, 1)
        all_nodes: list = sorted(
            set(node for hyperedge in hypergraph.values() for node in hyperedge)
        )
        if verbose:
            print(f"we have {len(all_nodes)} nodes")

        labels: np.ndarray = hg["labels"]
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        dataset: dict = {
            "hypergraph": hypergraph,
            "features": features,
            "labels": labels,
            "n": features.shape[0],
        }

        # Add encodings for random walks, Laplacian, and curvature
        # they each contain only one element (one hg) at the end
        list_hgs_with_encodings: list = []


        if encodings_to_compute == "RW":
            # add the encodings
            list_hgs_with_encodings.extend(
                self.compute_random_walks(dataset, verbose, random_walk_type)
            )
        elif encodings_to_compute == "Laplacian":
            list_hgs_with_encodings.extend(
                self.compute_laplacian(dataset, verbose, laplacian_type)
            )
        elif encodings_to_compute == "LCP":
            list_hgs_with_encodings.extend(
                self.compute_lcp(dataset, verbose, curvature_type)
            )
        elif encodings_to_compute == "LDP":
            list_hgs_with_encodings.extend(
                self.compute_ldp(dataset, verbose)
            )

        # Get the specific encoding type and key
        encoding_subtype: str = {
            "Laplacian": laplacian_type,
            "LCP": curvature_type,
            "RW": random_walk_type,
            "LDP": "LDP",
        }[encodings_to_compute]

        encoding_key: str = ENCODING_MAP[encodings_to_compute][encoding_subtype]

        encoding_map: dict = {encoding_key: list_hgs_with_encodings}

        for encoding_type, encoding_list in encoding_map.items():
            save_file = (
                f"{lukas_file}_with_encodings_{encoding_type}_count_{count}.pickle"
            )
            with open(
                os.path.join(self.individual_files_dir, save_file), "wb"
            ) as handle:
                pickle.dump(encoding_list, handle)

        return list_hgs_with_encodings

    def _process_file(
        self,
        lukas_file: str,
        encodings_to_compute: str = "LDP",
        laplacian_type: str = "Hodge",
        random_walk_type: str = "WE",
        curvature_type: str = "FRC",
    ) -> tuple[list, list, list, list, list, list, list, list]:
        """Processes one file at a time.

        Used for parrallel computations.
        THe files are imdb, reddit and collab.

        Args:
            lukas_file:
                the file
        """
        list_hgs_with_encodings_file: list[dict] = []

        with open(os.path.join(self.d, f"{lukas_file}.pickle"), "rb") as handle:
            # list of hypergraphs
            hypergraphs: list[dict] = pickle.load(handle)
            print(f"The file {lukas_file} contains {len(hypergraphs)} hypergraphs")
            results = []
            with mp.Pool() as pool:
                for count, hg in enumerate(hypergraphs):
                    result = pool.apply_async(
                        self._process_hypergraph,
                        (hg, lukas_file, count),
                    )
                    results.append(result)
                pool.close()
                pool.join()

        # Retrieve and accumulate results
        for result in results:
            list_hgs_with_encodings = result.get()

            list_hgs_with_encodings_file.extend(list_hgs_with_encodings)


        # Get the specific encoding type and key
        encoding_subtype: str = {
            "Laplacian": laplacian_type,
            "LCP": curvature_type,
            "RW": random_walk_type,
            "LDP": "LDP",
        }[encodings_to_compute]

        encoding_key: str = ENCODING_MAP[encodings_to_compute][encoding_subtype]s

        # Save each accumulated encoding list as a separate pickle file
        accumulated_encodings = {encoding_key: list_hgs_with_encodings_file}

        for encoding_type, encoding_list in accumulated_encodings.items():
            combined_file = f"{lukas_file}_with_encodings_{encoding_type}.pickle"
            with open(os.path.join(self.d, combined_file), "wb") as handle:
                pickle.dump(encoding_list, handle)
                print(f"Saved combined encodings to {combined_file}")
                print(
                    f"Saved combined encodings to {combined_file} with {len(encoding_list)} elements"
                )

        return list_hgs_with_encodings_file
