"""File used to save the encodings for lukas

Will just save in the same format Lukas provided to me.ie a list of dict
And now the "features" field of every dict will have been updated with the features

"""

import inspect
import csv
import os
import pickle
import warnings
from encodings_hnns.encodings import HypergraphEncodings
from encodings_hnns.laplacians import DisconnectedError

# import hypernetx as hnx
import numpy as np

# necessary for pickle.load
import scipy.sparse as sp

warnings.simplefilter("ignore")


class encodings_saver(object):
    """Parses data"""

    def __init__(self, data: str, dataset: str) -> None:
        """Initialises the data directory

        Arguments:
            data:
                coauthorship/cocitation
            dataset:
                cora/dblp/acm for coauthorship and cora/citeseer/pubmed for cocitation
        """

        current = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        )
        current = os.path.dirname(os.path.dirname(current))
        # Makes the path
        if data == "coauthorship" or data == "cocitation":
            self.d: str = os.path.join(current, "data", data, dataset)
        else:
            self.d: str = os.path.join(current, "data", data)
        self.data, self.dataset = data, dataset

    def compute_encodings(self):
        """Returns a dataset specific function to compute the
        encodings on the data added by Lukas

        Returns:
            TODO
        """

        name: str = "_compute_encodings"
        function = getattr(self, name, lambda: {})
        return function()

    def _compute_encodings(self, verbose: bool = True) -> dict:
        """Computes the encodings on the data"""
        hgencodings = HypergraphEncodings()

        list_files: list[str] = [
            "reddit_hypergraphs",
            "imdb_hypergraphs",
            "collab_hypergraphs",
        ]

        for lukas_file in list_files:
            list_hgs_rw_EE: list[dict] = []
            list_hgs_rw_EN: list[dict] = []
            list_hgs_rw_WE: list[dict] = []
            list_hgs_lape_hodge: list[dict] = []
            list_hgs_lape_normalized: list[dict] = []
            list_hgs_orc: list[dict] = []
            list_hgs_frc: list[dict] = []
            list_hgs_ldp: list[dict] = []
            with open(os.path.join(self.d, f"{lukas_file}.pickle"), "rb") as handle:
                # list of hypergraphs
                hypergraphs: list[dict] = pickle.load(handle)
                # loop through the list of hypergraphs
                count = 0
                for hg in hypergraphs[1:]:
                    print(lukas_file)
                    print(count)
                    # construct the hypergraph object in the same way we are used to
                    hypergraph: dict = hg["hypergraph"]

                    with open("debug.csv", "w", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        for key, value in hypergraph.items():
                            # If the value is a list, you may want to expand it into multiple columns
                            if isinstance(value, list):
                                writer.writerow([key] + value)
                            else:
                                writer.writerow([key, value])

                    features: np.ndarray = hg["features"]
                    all_nodes: list = sorted(
                        set(
                            node
                            for hyperedge in hypergraph.values()
                            for node in hyperedge
                        )
                    )
                    print(f"we have {len(all_nodes)} nodes")
                    features_shapes = features.shape
                    print(f"The features have shape {features.shape}")
                    labels: np.ndarray = hg["labels"]
                    dataset: dict = {
                        "hypergraph": hypergraph,
                        "features": features,
                        "labels": labels,
                        "n": features.shape[0],
                    }
                    # add the encodings
                    try:
                        for random_walk_type in ["EE", "EN", "WE"]:
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
                            features_shapes_with_encodings = dataset_copy[
                                "features"
                            ].shape
                            assert features_shapes != features_shapes_with_encodings
                            if random_walk_type == "WE":
                                list_hgs_rw_WE.append(dataset_copy)
                            elif random_walk_type == "EE":
                                list_hgs_rw_EE.append(dataset_copy)
                            elif random_walk_type == "EN":
                                list_hgs_rw_EN.append(dataset_copy)
                    except DisconnectedError as e:
                        print(f"Error: {e}")
                        list_hgs_rw_WE.append(dataset_copy)
                        list_hgs_rw_EE.append(dataset_copy)
                        list_hgs_rw_EN.append(dataset_copy)

                    try:
                        for laplacian_type in ["Hodge", "Normalized"]:
                            dataset_copy = dataset.copy()
                            dataset_copy = hgencodings.add_laplacian_encodings(
                                dataset_copy,
                                type=laplacian_type,
                                normalized=True,
                                dataset_name=None,
                            )
                            if laplacian_type == "Hodge":
                                list_hgs_lape_hodge.append(dataset_copy)
                            elif laplacian_type == "Normalized":
                                list_hgs_lape_normalized.append(dataset_copy)
                    except DisconnectedError as e:
                        print(f"Error: {e}")
                        list_hgs_lape_hodge.append(dataset_copy)
                        list_hgs_lape_normalized.append(dataset_copy)
                    try:
                        for curvature_type in ["FRC", "ORC"]:
                            dataset_copy = dataset.copy()
                            dataset_copy = hgencodings.add_curvature_encodings(
                                dataset_copy,
                                type=curvature_type,
                                normalized=True,
                                dataset_name=None,
                            )
                            if curvature_type == "ORC":
                                list_hgs_orc.append(dataset_copy)
                            elif curvature_type == "FRC":
                                list_hgs_frc.append(dataset_copy)
                    except DisconnectedError as e:
                        print(f"Error: {e}")
                        list_hgs_orc.append(dataset_copy)
                        list_hgs_frc.append(dataset_copy)
                    try:
                        dataset_copy = dataset.copy()
                        dataset_copy = hgencodings.add_degree_encodings(
                            dataset_copy, normalized=True, dataset_name=None
                        )
                        list_hgs_ldp.append(dataset_copy)
                    except DisconnectedError as e:
                        print(f"Error: {e}")
                        list_hgs_ldp.append(dataset_copy)

                    count += 1

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

            # Loop through the encoding map and save each list to a file
            for encoding_type, encoding_list in encoding_map.items():
                save_file = f"{lukas_file}_with_encodings_{encoding_type}.pickle"
                with open(os.path.join(self.d, save_file), "wb") as handle:
                    pickle.dump(encoding_list, handle)


# Run
if __name__ == "__main__":
    lukas = True
    if lukas:
        data_type = "hypergraph_classification_datasets"
        dataset_name = "reddit_hypergraphs"  # does not matter

        # Creates an instance of the encodings_saver class
        encodings_saver_instance = encodings_saver(data_type, dataset_name)
        # parse calls load_data
        parsed_data = encodings_saver_instance.compute_encodings()
