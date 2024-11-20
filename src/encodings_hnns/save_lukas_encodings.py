"""File used to save the encodings for lukas

Will just save in the same format Lukas provided to me.ie a list of dict
And now the "features" field of every dict will have been updated with the features

"""

import csv
import inspect
import multiprocessing as mp
import os
import pickle
import warnings

# import hypernetx as hnx
import numpy as np

# necessary for pickle.load
import scipy.sparse as sp
from tqdm import tqdm
import torch

from encodings_hnns.encodings import HypergraphEncodings
from encodings_hnns.laplacians import DisconnectedError

warnings.simplefilter("ignore")


def _convert_to_hypergraph(dataset: torch.Tensor) -> dict:
    """Converts a single graph to hypergraph format.

    Args:
        dataset: Single graph data

    Returns:
        Dictionary containing hypergraph data
    """
    x = dataset[0]  # node features
    edge_attr = dataset[1]  # edge features
    edge_index = dataset[2]  # connectivity
    y = dataset[3]  # labels

    # Convert to hypergraph format
    hypergraph = {}
    for i in range(edge_index.shape[1]):
        # Create hyperedge from each edge
        hypergraph[f"e_{i}"] = edge_index[:, i].tolist()

    return {
        "hypergraph": hypergraph,
        "features": x.numpy(),
        "labels": y.numpy() if isinstance(y, torch.Tensor) else y,
        "n": x.shape[0],
    }


def load_and_convert_lrgb_datasets(base_path: str, dataset_name: str) -> list:
    """Loads and converts LRGB datasets into the required format.

    Args:
        base_path:
            Path to the LRGB datasets directory
        dataset_name:
            Name of the dataset (e.g., 'peptidesstruct')

    Returns:
        List of converted Data objects
    """
    print(f"Loading {dataset_name} from {base_path}...")
    # Load train, val, and test sets
    train_data = torch.load(os.path.join(base_path, dataset_name, "train.pt"))
    val_data = torch.load(os.path.join(base_path, dataset_name, "val.pt"))
    test_data = torch.load(os.path.join(base_path, dataset_name, "test.pt"))

    print(f"Train data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    # print the first few elements
    print(f"Train data first few elements: {train_data[:5]}")
    print(f"Val data first few elements: {val_data[:5]}")
    print(f"Test data first few elements: {test_data[:5]}")
    # Convert all datasets
    converted_data = []
    for dataset in [train_data, val_data, test_data]:
        for i in range(len(dataset)):
            converted_data.append(_convert_to_hypergraph(dataset[i]))
    return converted_data


class encodings_saver(object):
    """Parses data"""

    def __init__(self, data: str) -> None:
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
            self.d: str = os.path.join(current, "data", data)
        else:
            self.d: str = os.path.join(current, "data", data)
        self.data = data

    def compute_encodings(self):
        """Returns a dataset specific function to compute the
        encodings on the data added by Lukas

        Returns:
            TODO
        """

        name: str = "_compute_encodings"
        function = getattr(self, name, lambda: {})
        return function()

    def _process_hypergraph(
        self,
        hg,
        lukas_file: str,
        count: int,
        verbose: bool = False,
    ) -> tuple[list, list, list, list, list, list, list, list]:
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
        # construct the hypergraph object in the same way we are used to
        hypergraph: dict = hg["hypergraph"]
        features: np.ndarray = hg["features"]
        all_nodes: list = sorted(
            set(node for hyperedge in hypergraph.values() for node in hyperedge)
        )
        if verbose:
            print(f"we have {len(all_nodes)} nodes")
        features_shapes = features.shape
        if verbose:
            print(f"The features have shape {features.shape}")
        labels: np.ndarray = hg["labels"]
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
                    type=laplacian_type,
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
                    type=curvature_type,
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
                f"{lukas_file}_with_encodings_{encoding_type}_count_{count}.pickle"
            )
            with open(
                os.path.join(self.d, "individual_files", save_file), "wb"
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

    def _process_file(
        self, lukas_file: str
    ) -> tuple[list, list, list, list, list, list, list, list]:
        """Processes one file at a time.

        Used for parrallel computations.
        THe files are imdb, reddit and collab.

        Args:
            lukas_file:
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
            (rw_EE, rw_EN, rw_WE, lape_hodge, lape_normalized, orc, frc, ldp) = (
                result.get()
            )

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
            combined_file = f"{lukas_file}_with_encodings_{encoding_type}.pickle"
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

        # for result in results:
        #     list_hgs_rw_EE.extend(result[0])
        #     list_hgs_rw_EN.extend(result[1])
        #     list_hgs_rw_WE.extend(result[2])
        #     list_hgs_lape_hodge.extend(result[3])
        #     list_hgs_lape_normalized.extend(result[4])
        #     list_hgs_orc.extend(result[5])
        #     list_hgs_frc.extend(result[6])
        #     list_hgs_ldp.extend(result[7])

        # encoding_map: dict = {
        #     "rw_EE": list_hgs_rw_EE,
        #     "rw_EN": list_hgs_rw_EN,
        #     "rw_WE": list_hgs_rw_WE,
        #     "lape_hodge": list_hgs_lape_hodge,
        #     "lape_normalized": list_hgs_lape_normalized,
        #     "orc": list_hgs_orc,
        #     "frc": list_hgs_frc,
        #     "ldp": list_hgs_ldp,
        #     # Add other mappings as needed
        # }

        # # Loop through the encoding map and save each list to a file
        # for encoding_type, encoding_list in encoding_map.items():
        #     save_file = f"{lukas_file}_with_encodings_{encoding_type}.pickle"
        #     with open(os.path.join(self.d, save_file), "wb") as handle:
        #         pickle.dump(encoding_list, handle)

        # # Return all lists
        # return (
        #     list_hgs_rw_EE,
        #     list_hgs_rw_EN,
        #     list_hgs_rw_WE,
        #     list_hgs_lape_hodge,
        #     list_hgs_lape_normalized,
        #     list_hgs_orc,
        #     list_hgs_frc,
        #     list_hgs_ldp,
        # )

    def _compute_encodings(self, verbose: bool = True) -> dict:
        """Computes the encodings on the data

        Args:
            verbose:
                whether to print more messages
        """

        list_files: list[str] = [
            "proteins_hypergraphs",
            "enzymes_hypergraphs",
            "mutag_hypergraphs",
            "imdb_hypergraphs",
            "collab_hypergraphs",
            "reddit_hypergraphs",
        ]

        all_results: dict = {}

        for lukas_file in list_files:
            results = self._process_file(lukas_file)
            all_results[lukas_file] = results

        return all_results


# Run
if __name__ == "__main__":
    lukas = False
    if lukas:
        data_type = "hypergraph_classification_datasets"
        # dataset_name = "reddit_hypergraphs"  # does not matter

        # Creates an instance of the encodings_saver class
        encodings_saver_instance = encodings_saver(data_type)
        # parse calls load_data
        parsed_data = encodings_saver_instance.compute_encodings()

    cluster = True
    if cluster:
        base_path = "/n/holyscratch01/mweber_lab/lrgb_datasets"
        datasets = [
            "peptidesstruct",
        ]

    for dataset_name in datasets:
        print(f"Processing {dataset_name}...")
        converted_dataset = load_and_convert_lrgb_datasets(base_path, dataset_name)

        assert False

        # Save the converted dataset
        save_path = os.path.join("data", "graph_classification_datasets")
        os.makedirs(save_path, exist_ok=True)

        save_file = os.path.join(save_path, f"{dataset_name}_hypergraphs.pickle")
        with open(save_file, "wb") as handle:
            pickle.dump(converted_dataset, handle)
        print(f"Saved {len(converted_dataset)} graphs to {save_file}")
