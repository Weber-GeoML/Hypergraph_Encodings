#!/usr/bin/env python3
"""Script to analyze the structure of coauthorship and cocitation data.

THIS IS AN EDA SCRIPT.

This script loads and examines the data structure for:
- coauthorship: cora, dblp
- cocitation: citeseer, cora, pubmed

It prints examples and statistics to understand the data format.

These files have:

Features: Sparse matrices (scipy.sparse.csr.csr_matrix)
Hypergraph: Dictionary with hyperedges
Labels: List of node labels
Splits: Dictionary with 'train' and 'test' splits (10 different splits)
"""

import os
import pickle
import warnings
from typing import Any

import numpy as np

warnings.simplefilter("ignore")


def load_pickle_file(file_path: str) -> Any:
    """Load a pickle file and return its contents.

    Args:
        file_path: Path to the pickle file

    Returns:
        Contents of the pickle file
    """
    try:
        with open(file_path, "rb") as handle:
            return pickle.load(handle)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def analyze_dataset(dataset_path: str, dataset_name: str) -> None:
    """Analyze a single dataset and print its structure.

    Args:
        dataset_path: Path to the dataset directory
        dataset_name: Name of the dataset
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING DATASET: {dataset_name.upper()}")
    print(f"{'='*60}")

    # Load features
    features_path = os.path.join(dataset_path, "features.pickle")
    features = load_pickle_file(features_path)
    if features is not None:
        print(f"Features type: {type(features)}")
        if isinstance(features, np.ndarray):
            print(f"Features shape: {features.shape}")
            print(f"Features dtype: {features.dtype}")
            print(f"Features sample (first 5 elements): {features[:5]}")
        else:
            print(f"Features content (first 100 chars): {str(features)[:100]}...")

    # Load hypergraph
    hypergraph_path = os.path.join(dataset_path, "hypergraph.pickle")
    hypergraph = load_pickle_file(hypergraph_path)
    if hypergraph is not None:
        print(f"\nHypergraph type: {type(hypergraph)}")
        if isinstance(hypergraph, dict):
            print(f"Number of hyperedges: {len(hypergraph)}")
            print("Sample hyperedges (first 3):")
            for i, (key, value) in enumerate(list(hypergraph.items())[:3]):
                print(f"  {key}: {value}")

            # Analyze hyperedge sizes
            edge_sizes = [len(edge) for edge in hypergraph.values()]
            print("Hyperedge size statistics:")
            print(f"  Min: {min(edge_sizes)}")
            print(f"  Max: {max(edge_sizes)}")
            print(f"  Mean: {np.mean(edge_sizes):.2f}")
            print(f"  Median: {np.median(edge_sizes):.2f}")

            # Get all unique nodes
            all_nodes = set()
            for edge in hypergraph.values():
                all_nodes.update(edge)
            print(f"Total unique nodes: {len(all_nodes)}")

    # Load labels
    labels_path = os.path.join(dataset_path, "labels.pickle")
    labels = load_pickle_file(labels_path)
    if labels is not None:
        print(f"\nLabels type: {type(labels)}")
        if isinstance(labels, np.ndarray):
            print(f"Labels shape: {labels.shape}")
            print(f"Labels dtype: {labels.dtype}")
            print(f"Labels sample (first 10): {labels[:10]}")
            print(f"Unique labels: {np.unique(labels)}")
            print(f"Number of classes: {len(np.unique(labels))}")
        else:
            print(f"Labels content (first 100 chars): {str(labels)[:100]}...")

    # Load splits
    splits_dir = os.path.join(dataset_path, "splits")
    if os.path.exists(splits_dir):
        print(f"\nSplits directory: {splits_dir}")
        split_files = [f for f in os.listdir(splits_dir) if f.endswith(".pickle")]
        print(f"Number of split files: {len(split_files)}")

        # Analyze first split
        if split_files:
            first_split_path = os.path.join(splits_dir, split_files[0])
            split_data = load_pickle_file(first_split_path)
            if split_data is not None:
                print(f"First split ({split_files[0]}) type: {type(split_data)}")
                if isinstance(split_data, dict):
                    print(f"Split keys: {list(split_data.keys())}")
                    for key, value in split_data.items():
                        if isinstance(value, (list, np.ndarray)):
                            print(f"  {key}: {len(value)} elements")
                        else:
                            print(f"  {key}: {type(value)}")
                elif isinstance(split_data, (list, tuple)):
                    print(f"Split contains {len(split_data)} elements")
                    if len(split_data) >= 3:
                        print(f"  Train: {len(split_data[0])} elements")
                        print(f"  Val: {len(split_data[1])} elements")
                        print(f"  Test: {len(split_data[2])} elements")


def main() -> None:
    """Main function to analyze all datasets."""
    print("ANALYZING COAUTHORSHIP AND COCITATION DATASETS")
    print("=" * 60)

    # Get the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, "data")

    print(f"Data directory: {data_dir}")

    # Analyze coauthorship datasets
    coauthorship_dir = os.path.join(data_dir, "coauthorship")
    if os.path.exists(coauthorship_dir):
        print(f"\nFound coauthorship directory: {coauthorship_dir}")
        for dataset in ["cora", "dblp"]:
            dataset_path = os.path.join(coauthorship_dir, dataset)
            if os.path.exists(dataset_path):
                analyze_dataset(dataset_path, f"coauthorship_{dataset}")
            else:
                print(f"Dataset {dataset} not found in coauthorship")
    else:
        print("Coauthorship directory not found")

    # Analyze cocitation datasets
    cocitation_dir = os.path.join(data_dir, "cocitation")
    if os.path.exists(cocitation_dir):
        print(f"\nFound cocitation directory: {cocitation_dir}")
        for dataset in ["citeseer", "cora", "pubmed"]:
            dataset_path = os.path.join(cocitation_dir, dataset)
            if os.path.exists(dataset_path):
                analyze_dataset(dataset_path, f"cocitation_{dataset}")
            else:
                print(f"Dataset {dataset} not found in cocitation")
    else:
        print("Cocitation directory not found")


if __name__ == "__main__":
    main()
