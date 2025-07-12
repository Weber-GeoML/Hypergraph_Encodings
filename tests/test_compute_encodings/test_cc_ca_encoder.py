#!/usr/bin/env python3
"""Test script for the EncodingsSaverForCCCA class.

This script tests the new class for processing coauthorship and cocitation datasets.
"""

import sys
import os
import warnings
from typing import Dict, Any
from compute_encodings.encoding_saver_cc_ca import EncodingsSaverForCCCA

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)


warnings.simplefilter("ignore")


def test_coauthorship_datasets() -> None:
    """Test processing coauthorship datasets."""
    print("Testing coauthorship datasets...")

    try:
        # Initialize the encoder
        encoder = EncodingsSaverForCCCA("coauthorship")

        # Get dataset info
        for dataset_name in ["cora", "dblp"]:
            try:
                info = encoder.get_dataset_info(dataset_name)
                print(f"\nDataset info for {dataset_name}:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
            except Exception as e:
                print(f"Error getting info for {dataset_name}: {e}")

        # Test processing a single dataset (smaller one first)
        print("\nTesting single dataset processing...")
        try:
            results = encoder._process_single_dataset(
                "cora", verbose=True, test_mode=True
            )
            print("Successfully processed cora dataset")
            print(f"Number of splits processed: {len(results)}")

            # Print some details about the results
            for split_name, split_results in results.items():
                print(f"  Split {split_name}: {len(split_results)} encoding types")
        except Exception as e:
            print(f"Error processing cora dataset: {e}")

    except Exception as e:
        print(f"Error initializing coauthorship encoder: {e}")


def test_cocitation_datasets() -> None:
    """Test processing cocitation datasets."""
    print("\nTesting cocitation datasets...")

    try:
        # Initialize the encoder
        encoder = EncodingsSaverForCCCA("cocitation")

        # Get dataset info
        for dataset_name in ["citeseer", "cora", "pubmed"]:
            try:
                info = encoder.get_dataset_info(dataset_name)
                print(f"\nDataset info for {dataset_name}:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
            except Exception as e:
                print(f"Error getting info for {dataset_name}: {e}")

        # Test processing a single dataset (smaller one first)
        print("\nTesting single dataset processing...")
        try:
            results = encoder._process_single_dataset(
                "citeseer", verbose=True, test_mode=True
            )
            print("Successfully processed citeseer dataset")
            print(f"Number of splits processed: {len(results)}")

            # Print some details about the results
            for split_name, split_results in results.items():
                print(f"  Split {split_name}: {len(split_results)} encoding types")
        except Exception as e:
            print(f"Error processing citeseer dataset: {e}")

    except Exception as e:
        print(f"Error initializing cocitation encoder: {e}")


def test_all_datasets() -> None:
    """Test processing all available datasets for both types."""
    print("\nTesting all datasets processing...")

    # Test coauthorship datasets
    print("\n--- Coauthorship Datasets ---")
    coauthorship_encoder = EncodingsSaverForCCCA("coauthorship")
    coauthorship_results: Dict[str, Any] = {}

    for dataset_name in coauthorship_encoder.available_datasets["coauthorship"]:
        print(f"\nProcessing coauthorship dataset: {dataset_name}")
        try:
            results = coauthorship_encoder._process_single_dataset(
                dataset_name, verbose=False, test_mode=True
            )
            coauthorship_results[dataset_name] = results
            print(f"  ✓ Successfully processed {dataset_name} ({len(results)} splits)")
        except Exception as e:
            print(f"  ✗ Error processing {dataset_name}: {e}")

    # Test cocitation datasets
    print("\n--- Cocitation Datasets ---")
    cocitation_encoder = EncodingsSaverForCCCA("cocitation")
    cocitation_results: Dict[str, Any] = {}

    for dataset_name in cocitation_encoder.available_datasets["cocitation"]:
        print(f"\nProcessing cocitation dataset: {dataset_name}")
        try:
            results = cocitation_encoder._process_single_dataset(
                dataset_name, verbose=False, test_mode=True
            )
            cocitation_results[dataset_name] = results
            print(f"  ✓ Successfully processed {dataset_name} ({len(results)} splits)")
        except Exception as e:
            print(f"  ✗ Error processing {dataset_name}: {e}")

    # Summary
    print("\n--- Summary ---")
    print(
        f"Coauthorship datasets processed: {len(coauthorship_results)}/{len(coauthorship_encoder.available_datasets['coauthorship'])}"
    )
    print(
        f"Cocitation datasets processed: {len(cocitation_results)}/{len(cocitation_encoder.available_datasets['cocitation'])}"
    )


def main() -> None:
    """Main test function."""
    print("TESTING ENCODINGSAVERFORCCCA CLASS")
    print("=" * 50)

    # Test individual dataset types
    test_coauthorship_datasets()
    test_cocitation_datasets()

    # Test all datasets
    test_all_datasets()

    print("\n" + "=" * 50)
    print("TESTING COMPLETED")


if __name__ == "__main__":
    main()
