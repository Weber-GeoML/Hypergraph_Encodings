# -*- coding: utf-8 -*-
"""Script for running hyperparameter tuning experiments"""

import os
import sys
import time
import datetime
import pandas as pd
import argparse
from typing import List, Dict, Any

# Add necessary paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "unignn"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from hgnn_config import (
    get_hyperparameter_tuning_configs,
    ENCODING_TYPES,
)
from hgnn_m3 import run_experiments_for_encoding
import torch


def run_hyperparameter_tuning(
    data_type: str = "cocitation",
    dataset_name: str = "cora",
    encoding_type: str = "none",
) -> List[Dict[str, Any]]:
    """
    Run hyperparameter tuning for a specific dataset and encoding.

    Args:
        data_type: Type of data (cocitation, coauthorship)
        dataset_name: Name of dataset (cora, citeseer, etc.)
        encoding_type: Type of encoding to test

    Returns:
        List of results for each hyperparameter configuration
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running hyperparameter tuning on {device}")
    print(f"Dataset: {data_type}/{dataset_name}")
    print(f"Encoding: {encoding_type}")

    # Get all hyperparameter configurations
    configs = get_hyperparameter_tuning_configs()
    print(f"Testing {len(configs)} hyperparameter configurations")

    results = []

    for i, config in enumerate(configs):
        print(f"\n--- Configuration {i+1}/{len(configs)} ---")
        print(f"Config: {config.to_dict()}")

        try:
            result = run_experiments_for_encoding(
                data_type, dataset_name, encoding_type, config, device
            )
            results.append(result)

        except Exception as e:
            print(f"Configuration {i+1} failed: {e}")
            continue

    return results


def main():
    """Main function for hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Run HGNN hyperparameter tuning")
    parser.add_argument(
        "--data_type",
        type=str,
        default="cocitation",
        choices=["cocitation", "coauthorship"],
        help="Type of data (cocitation, coauthorship)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cora",
        help="Name of dataset (cora, citeseer, pubmed, dblp)",
    )
    parser.add_argument(
        "--encoding_type",
        type=str,
        default="none",
        choices=ENCODING_TYPES,
        help="Type of encoding to test",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("HGNN Hyperparameter Tuning")
    print("=" * 80)
    print(f"Target: {args.data_type}/{args.dataset_name} with {args.encoding_type}")

    # Run hyperparameter tuning for the specified target
    results = run_hyperparameter_tuning(
        args.data_type, args.dataset_name, args.encoding_type
    )

    # Create summary
    if results:
        df = pd.DataFrame(results)

        # Find best configuration
        if len(df) > 0:
            best_idx = df["mean_test_acc_best_val"].idxmax()
            best_result = df.loc[best_idx]

            print("\n" + "=" * 60)
            print("BEST CONFIGURATION")
            print("=" * 60)
            print(f"Dataset: {args.data_type}/{args.dataset_name}")
            print(f"Encoding: {args.encoding_type}")
            print(
                f"Best accuracy: {best_result['mean_test_acc_best_val']:.4f} ± {best_result['std_test_acc_best_val']:.4f}"
            )
            print(f"Best config: {best_result['config']}")

        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"hyperparameter_tuning_{args.data_type}_{args.dataset_name}_{args.encoding_type}_{timestamp}.csv"
        df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
