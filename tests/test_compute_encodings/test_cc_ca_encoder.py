#!/usr/bin/env python3
"""Test script for the EncodingsSaverForCCCA class.

This script tests the new class for processing coauthorship and cocitation datasets.
"""

import sys
import os
import warnings

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

from compute_encodings.encoding_saver_cc_ca import EncodingsSaverForCCCA

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
            results = encoder.compute_encodings_for_dataset(
                "cora", verbose=True, test_mode=True
            )
            print(f"Successfully processed cora dataset")
            print(f"Number of splits processed: {len(results)}")
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
            results = encoder.compute_encodings_for_dataset(
                "citeseer", verbose=True, test_mode=True
            )
            print(f"Successfully processed citeseer dataset")
            print(f"Number of splits processed: {len(results)}")
        except Exception as e:
            print(f"Error processing citeseer dataset: {e}")

    except Exception as e:
        print(f"Error initializing cocitation encoder: {e}")


def main() -> None:
    """Main test function."""
    print("TESTING ENCODINGSAVERFORCCCA CLASS")
    print("=" * 50)

    # Test coauthorship datasets
    test_coauthorship_datasets()

    # Test cocitation datasets
    test_cocitation_datasets()

    print("\n" + "=" * 50)
    print("TESTING COMPLETED")


if __name__ == "__main__":
    main()
