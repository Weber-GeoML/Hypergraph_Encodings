"""Class for saving encodings for coauthorship and cocitation datasets.

These datasets have a different structure than hypergraph classification datasets:
- Single hypergraph per dataset
- Sparse feature matrices
- Node-level classification task
- Multiple train/test splits

Eg:
.
├── features.pickle
├── hypergraph.pickle
├── labels.pickle
└── splits
    ├── 1.pickle
    ├── 10.pickle
    ├── 2.pickle
    ├── 3.pickle
    ├── 4.pickle
    ├── 5.pickle
    ├── 6.pickle
    ├── 7.pickle
    ├── 8.pickle
    └── 9.pickle

1 directory, 13 files

"""

import os
import pickle
import warnings
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.sparse import csr_matrix

from compute_encodings.base_class import EncodingsSaverBase

warnings.simplefilter("ignore")


class EncodingsSaverForCCCA(EncodingsSaverBase):
    """Specialized class for coauthorship and cocitation datasets.

    Handles datasets with:
    - Single hypergraph per dataset
    - Sparse feature matrices
    - Node-level classification
    - Multiple train/test splits
    """

    def __init__(self, data_type: str) -> None:
        """Initialize the encoder for coauthorship/cocitation datasets.

        Args:
            data_type: Either 'coauthorship' or 'cocitation'
        """
        super().__init__(data_type)
        self.data_type = data_type

        # Define available datasets for each type
        self.available_datasets = {
            "coauthorship": ["cora", "dblp"],
            "cocitation": ["citeseer", "cora", "pubmed"],
        }

        print(f"Initialized EncodingsSaverForCCCA for {data_type}")
        print(f"Available datasets: {self.available_datasets.get(data_type, [])}")

    def _load_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Load a single dataset with its features, hypergraph, labels, and splits.

        Args:
            dataset_name: Name of the dataset (e.g., 'cora', 'dblp')

        Returns:
            Dictionary containing the dataset data
        """
        dataset_path = os.path.join(self.d, dataset_name)

        # Load features (sparse matrix)
        features_path = os.path.join(dataset_path, "features.pickle")
        with open(features_path, "rb") as handle:
            features = pickle.load(handle)

        # Load hypergraph
        hypergraph_path = os.path.join(dataset_path, "hypergraph.pickle")
        with open(hypergraph_path, "rb") as handle:
            hypergraph = pickle.load(handle)

        # Load labels
        labels_path = os.path.join(dataset_path, "labels.pickle")
        with open(labels_path, "rb") as handle:
            labels = pickle.load(handle)

        # Load splits
        splits_dir = os.path.join(dataset_path, "splits")
        splits = {}
        for split_file in os.listdir(splits_dir):
            if split_file.endswith(".pickle"):
                split_path = os.path.join(splits_dir, split_file)
                with open(split_path, "rb") as handle:
                    split_data = pickle.load(handle)
                    splits[split_file.replace(".pickle", "")] = split_data

        return {
            "features": features,
            "hypergraph": hypergraph,
            "labels": np.array(labels),
            "splits": splits,
            "dataset_name": dataset_name,
        }

    def _convert_sparse_to_dense(self, sparse_matrix: csr_matrix) -> np.ndarray:
        """Convert sparse matrix to dense numpy array.

        Args:
            sparse_matrix: Sparse matrix to convert

        Returns:
            Dense numpy array
        """
        return sparse_matrix.toarray()

    def _process_single_dataset(
        self, dataset_name: str, verbose: bool = False, test_mode: bool = False
    ) -> Dict[str, Any]:
        """Process a single dataset and compute encodings.

        Args:
            dataset_name: Name of the dataset
            verbose: Whether to print verbose output
            test_mode: If True, limit hyperedges for testing

        Returns:
            Dictionary containing computed encodings
        """
        print(f"Processing dataset: {dataset_name}")

        # Load the dataset
        dataset_data = self._load_dataset(dataset_name)

        # Convert sparse features to dense
        features = self._convert_sparse_to_dense(dataset_data["features"])
        print(f"Features shape: {features.shape}")
        hypergraph = dataset_data["hypergraph"]
        print(f"Hypergraph has {len(hypergraph)} hyperedges")
        labels = dataset_data["labels"]
        print(f"Labels shape: {labels.shape}")

        if test_mode:
            # Limit hyperedges for testing (take first 5 hyperedges)
            hypergraph = dict(list(hypergraph.items())[:5])
            print(f"TEST MODE: Limited to {len(hypergraph)} hyperedges")

        if verbose:
            print(f"Features shape: {features.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Number of hyperedges: {len(hypergraph)}")

        # Create the dataset structure (ONE hypergraph per dataset)
        dataset = {
            "hypergraph": hypergraph,
            "features": features,
            "labels": labels,
            "n": features.shape[0],
        }

        # Compute ALL encodings using _process_hypergraph (like other encoders)
        print(f"Computing encodings for {dataset_name}...")
        encoded_results = self._process_hypergraph(
            dataset,
            f"{dataset_name}_with_encodings",
            0,  # Single hypergraph per dataset
            verbose=verbose,
        )

        # Save the encoded features using the same naming convention
        encoding_types = [
            "rw_EE",
            "rw_EN",
            "rw_WE",
            "lape_hodge",
            "lape_normalized",
            "orc",
            "frc",
            "ldp",
        ]

        for i, encoding_type in enumerate(encoding_types):
            if encoded_results[i]:  # Check if encoding was computed successfully
                # Get the encoded dataset (should be only one in the list)
                encoded_dataset = encoded_results[i][
                    0
                ]  # Take the first (and only) dataset

                # Save just the features with encodings
                features_with_encodings = encoded_dataset["features"]
                save_file = (
                    f"{dataset_name}_features_with_encodings_{encoding_type}.npy"
                )
                save_path = os.path.join(self.d, save_file)

                np.save(save_path, features_with_encodings)
                print(
                    f"Saved encoded features to {save_file} (shape: {features_with_encodings.shape})"
                )

        # Return the encoded results (same format as other encoders)
        return {
            "rw_EE": encoded_results[0],
            "rw_EN": encoded_results[1],
            "rw_WE": encoded_results[2],
            "lape_hodge": encoded_results[3],
            "lape_normalized": encoded_results[4],
            "orc": encoded_results[5],
            "frc": encoded_results[6],
            "ldp": encoded_results[7],
        }

    def compute_encodings(
        self, verbose: bool = False, test_mode: bool = False
    ) -> Dict[str, Any]:
        """Compute encodings for all datasets of the specified type.

        Args:
            verbose: Whether to print verbose output
            test_mode: If True, process only first 2 splits and limit hyperedges

        Returns:
            Dictionary containing all computed encodings
        """
        available_datasets = self.available_datasets.get(self.data_type, [])

        print(
            f"Computing encodings for {self.data_type} datasets: {available_datasets}"
        )

        all_results = {}
        for dataset_name in available_datasets:
            try:
                dataset_results = self._process_single_dataset(
                    dataset_name, verbose, test_mode
                )
                all_results[dataset_name] = dataset_results
                print(f"Completed processing {dataset_name}")
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                continue

        return all_results

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary containing dataset information
        """
        dataset_data = self._load_dataset(dataset_name)

        features = dataset_data["features"]
        hypergraph = dataset_data["hypergraph"]
        labels = dataset_data["labels"]
        splits = dataset_data["splits"]

        # Get all unique nodes
        all_nodes = set()
        for edge in hypergraph.values():
            all_nodes.update(edge)

        # Analyze hyperedge sizes
        edge_sizes = [len(edge) for edge in hypergraph.values()]

        return {
            "dataset_name": dataset_name,
            "data_type": self.data_type,
            "features_shape": features.shape,
            "features_dtype": str(features.dtype),
            "num_hyperedges": len(hypergraph),
            "num_nodes": len(all_nodes),
            "num_classes": len(np.unique(labels)),
            "edge_size_stats": {
                "min": min(edge_sizes),
                "max": max(edge_sizes),
                "mean": np.mean(edge_sizes),
                "median": np.median(edge_sizes),
            },
            "num_splits": len(splits),
            "split_names": list(splits.keys()),
        }
