"""Class for saving the encodings for LRGB datasets.


The Long Range Graph Benchmark (LRGB) is a collection of 5 graph learning
datasets that arguably require long-range reasoning to achieve strong
performance in a given task. The 5 datasets in this benchmark can be used to
prototype new models that can capture long range dependencies in graphs.

Dataset	        Domain	    Task
Peptides-func	Chemistry	Graph Classification
Peptides-struct	Chemistry	Graph Regression

"""

import os
import warnings
from typing import Any, Dict, List

import torch
from base_class import EncodingsSaverBase

warnings.simplefilter("ignore")


class EncodingsSaverLRGB(EncodingsSaverBase):
    """Parses data for LRGB datasets."""

    def __init__(self, data: str) -> None:
        """
        Args:
            data: Name of the dataset type.
        """
        super().__init__("graph_classification_datasets")
        self.data: str = data

    @staticmethod
    def _convert_to_hypergraph_lrgb(dataset: Any) -> Dict[str, Any]:
        """
        Converts a single graph to hypergraph format for LRGB datasets.

        Args:
            dataset: Single graph data.

        Returns:
            Dictionary containing hypergraph data.
        """
        X = dataset[0]  # node features
        print(f"X shape: {X.shape}")
        edge_attr = dataset[1]  # edge features
        print(f"edge_attr shape: {edge_attr.shape}")
        edge_index = dataset[2]  # connectivity
        print(f"edge_index shape: {edge_index.shape}")
        y = dataset[3]  # labels
        # Convert to hypergraph format
        hypergraph: Dict[str, List[int]] = {}
        for i in range(edge_index.shape[1]):
            # Create hyperedge from each edge
            hypergraph[f"e_{i}"] = edge_index[:, i].tolist()

        return {
            "hypergraph": hypergraph,
            "features": X.numpy(),
            "labels": y.numpy() if isinstance(y, torch.Tensor) else y,
            "n": X.shape[0],
        }

    @staticmethod
    def load_and_convert_lrgb_datasets(
        base_path: str, dataset_name: str
    ) -> List[Dict[str, Any]]:
        """
        Loads and converts LRGB datasets into the required format.

        Args:
            base_path:
                Path to the LRGB datasets directory
            dataset_name:
                Name of the dataset (e.g., 'peptidesstruct')

        Returns:
            List of converted Data objects.
        """
        print(f"Loading {dataset_name} from {base_path}...")
        # Load train, val, and test sets
        train_data = torch.load(os.path.join(base_path, dataset_name, "train.pt"))
        val_data = torch.load(os.path.join(base_path, dataset_name, "val.pt"))
        test_data = torch.load(os.path.join(base_path, dataset_name, "test.pt"))
        print(f"Train data length: {len(train_data)}")
        print(f"Val data length: {len(val_data)}")
        print(f"Test data length: {len(test_data)}")
        # print the first few elements
        if False:
            print(f"Train data first few elements: {train_data[:5]}")
            print(f"Val data first few elements: {val_data[:5]}")
            print(f"Test data first few elements: {test_data[:5]}")
        print("*" * 100)
        print(f"the type is {type(train_data[0])}")
        print("*" * 100)
        print(train_data[0])
        print(len(train_data[0]))
        # Convert all datasets
        converted_data: List[Dict[str, Any]] = []
        for dataset in [train_data, val_data, test_data]:
            for i in range(len(dataset)):
                converted_data.append(
                    EncodingsSaverLRGB._convert_to_hypergraph_lrgb(dataset[i])
                )
        return converted_data
