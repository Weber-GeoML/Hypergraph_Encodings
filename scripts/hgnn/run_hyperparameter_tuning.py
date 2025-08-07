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

from hgnn.hgnn_config import (
    get_hyperparameter_tuning_configs,
    get_dataset_specific_configs,
    ENCODING_TYPES,
    DATASET_CONFIGS,
)
from hgnn_m3 import run_experiments_for_encoding
import torch

# Force CPU usage to avoid CUDA issues
torch.cuda.is_available = lambda: False
device = torch.device("cpu")
print(f"Using device: {device}")

# Import wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


def run_hyperparameter_tuning(
    data_type: str = "cocitation",
    dataset_name: str = "cora",
    encoding_type: str = "none",
    wandb_enabled: bool = False,
    wandb_project: str = "hgnn-hyperparameter-tuning",
    wandb_entity: str = "weber-geoml-harvard-university",
) -> List[Dict[str, Any]]:
    """
    Run hyperparameter tuning for a specific dataset and encoding.

    Args:
        data_type: Type of data (cocitation, coauthorship)
        dataset_name: Name of dataset (cora, citeseer, etc.)
        encoding_type: Type of encoding to test
        wandb_enabled: Whether to enable wandb logging
        wandb_project: Wandb project name
        wandb_entity: Wandb entity name

    Returns:
        List of results for each hyperparameter configuration
    """
    print(f"Running hyperparameter tuning on {device}")
    print(f"Dataset: {data_type}/{dataset_name}")
    print(f"Encoding: {encoding_type}")

    # Initialize wandb if enabled
    if wandb_enabled and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=f"hgnn_tuning_{data_type}_{dataset_name}_{encoding_type}",
            config={
                "data_type": data_type,
                "dataset_name": dataset_name,
                "encoding_type": encoding_type,
                "device": str(device),
            },
        )
        print(f"Initialized wandb run: {wandb.run.name}")

    # Get all hyperparameter configurations
    configs = get_hyperparameter_tuning_configs()
    print(f"Testing {len(configs)} hyperparameter configurations")

    results = []

    for i, config in enumerate(configs):
        print(f"\n--- Configuration {i+1}/{len(configs)} ---")
        print(f"Config: {config.to_dict()}")

        # Log configuration to wandb
        if wandb_enabled and WANDB_AVAILABLE:
            wandb.log({"config": config.to_dict()}, step=i)

        try:
            result = run_experiments_for_encoding(
                data_type, dataset_name, encoding_type, config, device
            )
            results.append(result)

            # Log results to wandb
            if wandb_enabled and WANDB_AVAILABLE and result:
                wandb.log(
                    {
                        "config_index": i,
                        "mean_test_acc_best_val": result.get(
                            "mean_test_acc_best_val", 0.0
                        ),
                        "std_test_acc_best_val": result.get(
                            "std_test_acc_best_val", 0.0
                        ),
                        "mean_val_acc_best_val": result.get(
                            "mean_val_acc_best_val", 0.0
                        ),
                        "std_val_acc_best_val": result.get("std_val_acc_best_val", 0.0),
                        "best_learning_rate": result.get("config", {}).get(
                            "learning_rate", 0.0
                        ),
                        "best_hidden_dims": result.get("config", {}).get(
                            "hidden_dims", 0
                        ),
                        "best_dropout_rate": result.get("config", {}).get(
                            "dropout_rate", 0.0
                        ),
                        "best_weight_decay": result.get("config", {}).get(
                            "weight_decay", 0.0
                        ),
                    },
                    step=i,
                )

        except Exception as e:
            print(f"Configuration {i+1} failed: {e}")
            if wandb_enabled and WANDB_AVAILABLE:
                wandb.log({"error": str(e)}, step=i)
            continue

    # Finish wandb run
    if wandb_enabled and WANDB_AVAILABLE:
        wandb.finish()

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
    parser.add_argument(
        "--wandb_enabled",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="hgnn-hyperparameter-tuning",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="weber-geoml-harvard-university",
        help="Wandb entity name",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("HGNN Hyperparameter Tuning")
    print("=" * 80)
    print(f"Target: {args.data_type}/{args.dataset_name} with {args.encoding_type}")
    print(f"Wandb enabled: {args.wandb_enabled}")

    # Run hyperparameter tuning for the specified target
    results = run_hyperparameter_tuning(
        args.data_type,
        args.dataset_name,
        args.encoding_type,
        args.wandb_enabled,
        args.wandb_project,
        args.wandb_entity,
    )

    # Create summary
    if results:
        df = pd.DataFrame(results)

        # Find best configuration
        if len(df) > 0:
            best_idx = df["mean_test_acc_best_val"].idxmax()
            best_result = df.loc[best_idx]

            print(f"\n" + "=" * 60)
            print("BEST CONFIGURATION")
            print("=" * 60)
            print(f"Dataset: {args.data_type}/{args.dataset_name}")
            print(f"Encoding: {args.encoding_type}")
            print(
                f"Best accuracy: {best_result['mean_test_acc_best_val']:.4f} ± {best_result['std_test_acc_best_val']:.4f}"
            )
            print(f"Best config: {best_result['config']}")

        # Save results to lab directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/hyperparameter_tuning_{args.data_type}_{args.dataset_name}_{args.encoding_type}_{timestamp}.csv"
        df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
