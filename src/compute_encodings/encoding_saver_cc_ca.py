"""Class for saving encodings for coauthorship and cocitation datasets.

These datasets have a different structure than hypergraph classification datasets:
- Single hypergraph per dataset
- Sparse feature matrices
- Node-level classification task
- Multiple train/test splits
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
        self, dataset_name: str, verbose: bool = False
    ) -> Dict[str, Any]:
        """Process a single dataset and compute encodings for all splits.

        Args:
            dataset_name: Name of the dataset
            verbose: Whether to print verbose output

        Returns:
            Dictionary containing all computed encodings
        """
        print(f"Processing dataset: {dataset_name}")

        # Load the dataset
        dataset_data = self._load_dataset(dataset_name)

        # Convert sparse features to dense
        features = self._convert_sparse_to_dense(dataset_data["features"])
        hypergraph = dataset_data["hypergraph"]
        labels = dataset_data["labels"]
        splits = dataset_data["splits"]

        if verbose:
            print(f"Features shape: {features.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Number of hyperedges: {len(hypergraph)}")
            print(f"Number of splits: {len(splits)}")

        # Create the base dataset structure
        base_dataset = {
            "hypergraph": hypergraph,
            "features": features,
            "labels": labels,
            "n": features.shape[0],
        }

        # Process each split
        all_results = {}
        for split_name, split_data in splits.items():
            print(f"Processing split: {split_name}")

            # Create dataset for this split
            split_dataset = base_dataset.copy()

            # Process the split dataset
            split_results = self._process_hypergraph(
                split_dataset,
                f"{dataset_name}_{split_name}",
                0,  # Single hypergraph per split
                verbose=verbose,
            )

            all_results[split_name] = split_results

        return all_results

    def compute_encodings(self, verbose: bool = False) -> Dict[str, Any]:
        """Compute encodings for all datasets of the specified type.

        Args:
            verbose: Whether to print verbose output

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
                dataset_results = self._process_single_dataset(dataset_name, verbose)
                all_results[dataset_name] = dataset_results
                print(f"Completed processing {dataset_name}")
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                continue

        return all_results

    def compute_encodings_for_dataset(
        self, dataset_name: str, verbose: bool = False
    ) -> Dict[str, Any]:
        """Compute encodings for a specific dataset.

        Args:
            dataset_name: Name of the specific dataset
            verbose: Whether to print verbose output

        Returns:
            Dictionary containing computed encodings for the dataset
        """
        available_datasets = self.available_datasets.get(self.data_type, [])

        if dataset_name not in available_datasets:
            raise ValueError(
                f"Dataset {dataset_name} not available for {self.data_type}. "
                f"Available: {available_datasets}"
            )

        return self._process_single_dataset(dataset_name, verbose)

    def save_encodings_for_split(
        self,
        dataset_name: str,
        split_name: str,
        encodings: Tuple[List, ...],
        encoding_types: List[str],
    ) -> None:
        """Save encodings for a specific dataset and split.

        Args:
            dataset_name: Name of the dataset
            split_name: Name of the split
            encodings: Tuple of encoding lists
            encoding_types: List of encoding type names
        """
        # Create directory for this dataset
        dataset_dir = os.path.join(self.d, f"{dataset_name}_encodings")
        os.makedirs(dataset_dir, exist_ok=True)

        # Save each encoding type
        for encoding_type, encoding_list in zip(encoding_types, encodings):
            if encoding_list:  # Only save if we have encodings
                save_file = (
                    f"{dataset_name}_{split_name}_with_encodings_{encoding_type}.pickle"
                )
                save_path = os.path.join(dataset_dir, save_file)

                with open(save_path, "wb") as handle:
                    pickle.dump(encoding_list, handle)

                print(f"Saved {save_file} with {len(encoding_list)} elements")

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
