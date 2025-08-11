#!/usr/bin/env python3
"""Script to compute encodings the old way and save to computed_encodings/

This is for CC and CA datasets!
"""

import os
import pickle
import sys
import time
import warnings
from datetime import datetime

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from encodings_hnns.encodings import HypergraphEncodings

warnings.simplefilter("ignore")


def log_with_timestamp(message: str) -> None:
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()  # Ensure immediate output


def compute_encodings_old_way(dataset_name: str, data_type: str) -> None:
    """Compute encodings using the old approach and save to computed_encodings/.

    Args:
        dataset_name: Name of the dataset (e.g., 'cora', 'dblp')
        data_type: Either 'coauthorship' or 'cocitation'
    """
    start_time = time.time()
    log_with_timestamp(f"Starting processing of {data_type} dataset: {dataset_name}")
    log_with_timestamp(f"Current working directory: {os.getcwd()}")

    # Load the dataset
    dataset_path = os.path.join("data", data_type, dataset_name)
    log_with_timestamp(f"Loading dataset from: {dataset_path}")

    # Load features (sparse matrix)
    features_path = os.path.join(dataset_path, "features.pickle")
    log_with_timestamp(f"Loading features from: {features_path}")
    with open(features_path, "rb") as handle:
        features = pickle.load(handle)

    # Load hypergraph
    hypergraph_path = os.path.join(dataset_path, "hypergraph.pickle")
    log_with_timestamp(f"Loading hypergraph from: {hypergraph_path}")
    with open(hypergraph_path, "rb") as handle:
        hypergraph = pickle.load(handle)

    # Load labels
    labels_path = os.path.join(dataset_path, "labels.pickle")
    log_with_timestamp(f"Loading labels from: {labels_path}")
    with open(labels_path, "rb") as handle:
        labels = pickle.load(handle)

    # Convert sparse features to dense
    if hasattr(features, "toarray"):
        log_with_timestamp("Converting sparse features to dense")
        features = features.toarray()

    # Create dataset dict
    dataset = {
        "hypergraph": hypergraph,
        "features": features,
        "labels": labels,
        "n": features.shape[0],
    }

    log_with_timestamp(
        f"Dataset info: {features.shape[0]} nodes, {features.shape[1]} features, {len(hypergraph)} hyperedges"
    )

    # Initialize HypergraphEncodings
    log_with_timestamp("Initializing HypergraphEncodings")
    hgencodings = HypergraphEncodings()

    # Compute and save degree encodings
    log_with_timestamp("Computing degree encodings...")
    encoding_start = time.time()
    try:
        _ = hgencodings.add_degree_encodings(
            dataset.copy(),
            verbose=False,
            normalized=True,
            dataset_name=f"{data_type}_{dataset_name}",
        )
        encoding_time = time.time() - encoding_start
        log_with_timestamp(f"✓ Degree encodings saved (took {encoding_time:.2f}s)")
    except Exception as e:
        log_with_timestamp(f"✗ Error with degree encodings: {e}")
        import traceback

        log_with_timestamp(f"Traceback: {traceback.format_exc()}")

    # Compute and save random walk encodings
    for rw_type in ["EE", "EN", "WE"]:
        log_with_timestamp(f"Computing random walk encodings ({rw_type})...")
        encoding_start = time.time()
        try:
            _ = hgencodings.add_randowm_walks_encodings(
                dataset.copy(),
                rw_type=rw_type,
                k=20,
                normalized=True,
                dataset_name=f"{data_type}_{dataset_name}",
            )
            encoding_time = time.time() - encoding_start
            log_with_timestamp(
                f"✓ Random walk encodings ({rw_type}) saved (took {encoding_time:.2f}s)"
            )
        except Exception as e:
            log_with_timestamp(f"✗ Error with random walk encodings ({rw_type}): {e}")
            import traceback

            log_with_timestamp(f"Traceback: {traceback.format_exc()}")

    # Compute and save Laplacian encodings
    for laplacian_type in ["Hodge", "Normalized"]:
        log_with_timestamp(f"Computing Laplacian encodings ({laplacian_type})...")
        encoding_start = time.time()
        try:
            _ = hgencodings.add_laplacian_encodings(
                dataset.copy(),
                laplacian_type=laplacian_type,
                normalized=True,
                dataset_name=f"{data_type}_{dataset_name}",
            )
            encoding_time = time.time() - encoding_start
            log_with_timestamp(
                f"✓ Laplacian encodings ({laplacian_type}) saved (took {encoding_time:.2f}s)"
            )
        except Exception as e:
            log_with_timestamp(
                f"✗ Error with Laplacian encodings ({laplacian_type}): {e}"
            )
            import traceback

            log_with_timestamp(f"Traceback: {traceback.format_exc()}")

    # Compute and save curvature encodings
    for curvature_type in ["ORC", "FRC"]:
        log_with_timestamp(f"Computing curvature encodings ({curvature_type})...")
        encoding_start = time.time()
        try:
            _ = hgencodings.add_curvature_encodings(
                dataset.copy(),
                curvature_type=curvature_type,
                normalized=True,
                dataset_name=f"{data_type}_{dataset_name}",
            )
            encoding_time = time.time() - encoding_start
            log_with_timestamp(
                f"✓ Curvature encodings ({curvature_type}) saved (took {encoding_time:.2f}s)"
            )
        except Exception as e:
            log_with_timestamp(
                f"✗ Error with curvature encodings ({curvature_type}): {e}"
            )
            import traceback

            log_with_timestamp(f"Traceback: {traceback.format_exc()}")

    total_time = time.time() - start_time
    log_with_timestamp(
        f"Completed processing {data_type}/{dataset_name} in {total_time:.2f}s"
    )


def main() -> None:
    """Main function to compute encodings for all datasets."""
    main_start_time = time.time()
    log_with_timestamp("=" * 60)
    log_with_timestamp("Starting encoding computation using the old approach")
    log_with_timestamp("=" * 60)

    # Define datasets
    coauthorship_datasets = ["dblp"]  # ["dblp", "cora"]
    cocitation_datasets = []  # ["citeseer", "cora", "pubmed"]

    total_datasets = len(coauthorship_datasets) + len(cocitation_datasets)
    log_with_timestamp(f"Total datasets to process: {total_datasets}")

    processed_count = 0
    failed_count = 0

    # Process coauthorship datasets
    if coauthorship_datasets:
        log_with_timestamp("\n--- Coauthorship Datasets ---")
        for dataset_name in coauthorship_datasets:
            try:
                log_with_timestamp(
                    f"Processing coauthorship dataset {dataset_name} ({processed_count+1}/{total_datasets})"
                )
                compute_encodings_old_way(dataset_name, "coauthorship")
                processed_count += 1
                log_with_timestamp(f"✓ Successfully completed {dataset_name}")
            except Exception as e:
                failed_count += 1
                log_with_timestamp(
                    f"✗ Error processing coauthorship dataset {dataset_name}: {e}"
                )
                import traceback

                log_with_timestamp(f"Traceback: {traceback.format_exc()}")

    # Process cocitation datasets
    if cocitation_datasets:
        log_with_timestamp("\n--- Cocitation Datasets ---")
        for dataset_name in cocitation_datasets:
            try:
                log_with_timestamp(
                    f"Processing cocitation dataset {dataset_name} ({processed_count+1}/{total_datasets})"
                )
                compute_encodings_old_way(dataset_name, "cocitation")
                processed_count += 1
                log_with_timestamp(f"✓ Successfully completed {dataset_name}")
            except Exception as e:
                failed_count += 1
                log_with_timestamp(
                    f"✗ Error processing cocitation dataset {dataset_name}: {e}"
                )
                import traceback

                log_with_timestamp(f"Traceback: {traceback.format_exc()}")

    total_time = time.time() - main_start_time
    log_with_timestamp("\n" + "=" * 60)
    log_with_timestamp("ENCODING COMPUTATION SUMMARY")
    log_with_timestamp("=" * 60)
    log_with_timestamp(f"Total datasets processed: {processed_count}/{total_datasets}")
    log_with_timestamp(f"Failed datasets: {failed_count}")
    log_with_timestamp(
        f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )
    log_with_timestamp("Encoding computation completed!")

    # Exit with error code if any failures
    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
