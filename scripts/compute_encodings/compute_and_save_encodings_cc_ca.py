#!/usr/bin/env python3
"""Script to compute encodings the old way and save to computed_encodings/

This is dor CC and CA datasets!
"""

import os
import sys
import pickle
import warnings

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from encodings_hnns.encodings import HypergraphEncodings

warnings.simplefilter("ignore")


def compute_encodings_old_way(dataset_name: str, data_type: str) -> None:
    """Compute encodings using the old approach and save to computed_encodings/.

    Args:
        dataset_name: Name of the dataset (e.g., 'cora', 'dblp')
        data_type: Either 'coauthorship' or 'cocitation'
    """
    print(f"Processing {data_type} dataset: {dataset_name}")

    # Load the dataset
    dataset_path = os.path.join("data", data_type, dataset_name)

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

    # Convert sparse features to dense
    if hasattr(features, "toarray"):
        features = features.toarray()

    # Create dataset dict
    dataset = {
        "hypergraph": hypergraph,
        "features": features,
        "labels": labels,
        "n": features.shape[0],
    }

    # Initialize HypergraphEncodings
    hgencodings = HypergraphEncodings()

    # Compute and save degree encodings
    print("  Computing degree encodings...")
    try:
        degree_dataset = hgencodings.add_degree_encodings(
            dataset.copy(),
            verbose=False,
            normalized=True,
            dataset_name=f"{data_type}_{dataset_name}",
        )
        print("  ✓ Degree encodings saved")
    except Exception as e:
        print(f"  ✗ Error with degree encodings: {e}")

    # Compute and save random walk encodings
    for rw_type in ["EE", "EN", "WE"]:
        print(f"  Computing random walk encodings ({rw_type})...")
        try:
            rw_dataset = hgencodings.add_randowm_walks_encodings(
                dataset.copy(),
                rw_type=rw_type,
                k=20,
                normalized=True,
                dataset_name=f"{data_type}_{dataset_name}",
            )
            print(f"  ✓ Random walk encodings ({rw_type}) saved")
        except Exception as e:
            print(f"  ✗ Error with random walk encodings ({rw_type}): {e}")

    # Compute and save Laplacian encodings
    for laplacian_type in ["Hodge", "Normalized"]:
        print(f"  Computing Laplacian encodings ({laplacian_type})...")
        try:
            laplacian_dataset = hgencodings.add_laplacian_encodings(
                dataset.copy(),
                laplacian_type=laplacian_type,
                normalized=True,
                dataset_name=f"{data_type}_{dataset_name}",
            )
            print(f"  ✓ Laplacian encodings ({laplacian_type}) saved")
        except Exception as e:
            print(f"  ✗ Error with Laplacian encodings ({laplacian_type}): {e}")

    # Compute and save curvature encodings
    for curvature_type in ["ORC", "FRC"]:
        print(f"  Computing curvature encodings ({curvature_type})...")
        try:
            curvature_dataset = hgencodings.add_curvature_encodings(
                dataset.copy(),
                curvature_type=curvature_type,
                normalized=True,
                dataset_name=f"{data_type}_{dataset_name}",
            )
            print(f"  ✓ Curvature encodings ({curvature_type}) saved")
        except Exception as e:
            print(f"  ✗ Error with curvature encodings ({curvature_type}): {e}")


def main() -> None:
    """Main function to compute encodings for all datasets."""
    print("Computing encodings using the old approach...")
    print("=" * 60)

    # Define datasets
    coauthorship_datasets = ["cora", "dblp"]
    cocitation_datasets = ["citeseer", "cora", "pubmed"]

    # Process coauthorship datasets
    print("\n--- Coauthorship Datasets ---")
    for dataset_name in coauthorship_datasets:
        try:
            compute_encodings_old_way(dataset_name, "coauthorship")
        except Exception as e:
            print(f"Error processing coauthorship dataset {dataset_name}: {e}")

    # Process cocitation datasets
    print("\n--- Cocitation Datasets ---")
    for dataset_name in cocitation_datasets:
        try:
            compute_encodings_old_way(dataset_name, "cocitation")
        except Exception as e:
            print(f"Error processing cocitation dataset {dataset_name}: {e}")

    print("\n" + "=" * 60)
    print("Encoding computation completed!")


if __name__ == "__main__":
    main()
