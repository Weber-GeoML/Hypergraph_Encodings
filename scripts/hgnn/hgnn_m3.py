# -*- coding: utf-8 -*-
"""UniGNN-Compatible HGNN with 80 runs and pre-computed encodings"""

import datetime
import os
import time
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Tuple, Any, Optional
from tqdm import tqdm
import traceback
from pathlib import Path
import argparse
from dataclasses import dataclass

# Import wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# Import only the modules that don't depend on torch_sparse
from encodings_hnns.data_handling import load

# Import configuration
from hgnn.hgnn_config import (
    HGNNConfig,
    get_default_config,
    get_dataset_specific_configs,
    ENCODING_TYPES,
    DATASET_CONFIGS,
)
from hgnn.hgnn_architecture import HGNN

# Import best hyperparameters
from best_hyperparameters import get_best_hyperparameters

warnings.filterwarnings("ignore")
os.environ["TORCH"] = torch.__version__

# Force CPU usage to avoid CUDA issues
torch.cuda.is_available = lambda: False
device = torch.device("cpu")
print(f"Using device: {device}")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def accuracy(Z: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Computes the accuracy between prediction Z and true labels Y.
    M3-compatible version without torch_sparse dependency.
    """
    return 100 * Z.argmax(1).eq(Y).float().mean().item()


def load_precomputed_encoding(
    encoding_type: str, dataset_name: str
) -> Optional[Dict[str, Any]]:
    """
    Load pre-computed encoding from computed_encodings/ directory.

    Args:
        encoding_type: Type of encoding (degree, random_walk_EE, etc.)
        dataset_name: Dataset name (e.g., 'cocitation_cora')

    Returns:
        Loaded dataset dictionary or None if file doesn't exist
    """
    if encoding_type == "none":
        return None

    filename_map = {
        "degree": f"computed_encodings/{dataset_name}_degree_encodings_normalized_True.pkl",
        "random_walk_EE": f"computed_encodings/{dataset_name}_rw_encodings_EE_k_20_normalized_True.pkl",
        "random_walk_EN": f"computed_encodings/{dataset_name}_rw_encodings_EN_k_20_normalized_True.pkl",
        "random_walk_WE": f"computed_encodings/{dataset_name}_rw_encodings_WE_k_20_normalized_True.pkl",
        "laplacian_Hodge": f"computed_encodings/{dataset_name}_laplacian_encodings_Hodge_normalized_True.pkl",
        "laplacian_Normalized": f"computed_encodings/{dataset_name}_laplacian_encodings_Normalized_normalized_True.pkl",
        "curvature_FRC": f"computed_encodings/{dataset_name}_curvature_encodings_FRC_normalized_True.pkl",
        "curvature_ORC": f"computed_encodings/{dataset_name}_curvature_encodings_ORC_normalized_True.pkl",
    }

    if encoding_type not in filename_map:
        print(f"  Unknown encoding type: {encoding_type}")
        return None

    rel_path = filename_map[encoding_type]
    abs_path = Path(rel_path).resolve()
    cluster_path = (
        Path("/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/computed_encodings")
        / abs_path.name
    )

    print("  [ENC DEBUG] __file__      :", Path(__file__).resolve())
    print("  [ENC DEBUG] cwd           :", Path.cwd())
    print("  [ENC DEBUG] rel_path      :", rel_path)
    print("  [ENC DEBUG] abs_path      :", abs_path)
    print("  [ENC DEBUG] abs exists?   :", abs_path.exists())
    print("  [ENC DEBUG] cluster_path  :", cluster_path)
    print("  [ENC DEBUG] cluster exist?:", cluster_path.exists())

    target_path = abs_path if abs_path.exists() else cluster_path
    if target_path.exists():
        try:
            with open(target_path, "rb") as f:
                dataset = pickle.load(f)
            print(f"  Loaded pre-computed encoding from {target_path}")
            return dataset
        except Exception as e:
            print(f"  Error loading {target_path}: {e}")
            traceback.print_exc()
            return None
    else:
        print(f"  Pre-computed encoding not found at: {abs_path}")
        print(f"  Also not found at: {cluster_path}")
        return None


def load_base_data(args) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Load base data using UniGNN's data loading function.
    FIXED: Handle tuple return from load() function.
    """
    # Load data using UniGNN's function - returns (dataset, train, test)
    dataset, train_idx, test_idx = load(args)
    print(f"\n The split is {args.split}")

    # Extract data from dataset dictionary
    X = dataset["features"]
    Y = dataset["labels"]
    G = dataset["hypergraph"]

    # Convert to tensors if needed
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, dtype=torch.long)

    # FIX: Handle 2D one-hot encoded labels
    if Y.dim() == 2:
        # print(f"  Converting 2D labels {Y.shape} to 1D")
        Y = Y.argmax(dim=1)  # Convert one-hot to class indices

    # Validate labels are now 1D
    assert Y.dim() == 1, f"Labels must be 1D, got shape {Y.shape}"

    # Normalize features
    if args.normalize_features:
        print("Normalizing the features")
        X = F.normalize(X, p=2, dim=1)

    # Create hypergraph structure
    num_nodes = X.shape[0]
    num_classes = Y.max().item() + 1

    # Create hypergraph from graph edges
    if "edge_index" in G:
        edge_index = G["edge_index"]
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.cpu().numpy()

        # Group nodes by edges to create hyperedges
        hyperedges = {}
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src not in hyperedges:
                hyperedges[src] = []
            hyperedges[src].append(dst)

        # Add self-loops
        for node in range(num_nodes):
            if node not in hyperedges:
                hyperedges[node] = []
            hyperedges[node].append(node)

        G["hypergraph"] = hyperedges
    else:
        # Create default hyperedges (self-loops)
        G["hypergraph"] = {i: [i] for i in range(num_nodes)}

    # Add metadata
    G["num_features"] = X.shape[1]
    G["num_classes"] = num_classes

    # print(f"number of hyperedges is {len(G['hypergraph'])}")
    # print(
    #     f"The hypergraph {args.data}/{args.dataset} has {len(G['hypergraph'])} hyperedges where authors are hyperedges"
    # )

    # Calculate average hyperedge size
    total_nodes_in_edges = sum(len(nodes) for nodes in G["hypergraph"].values())
    avg_size = total_nodes_in_edges / len(G["hypergraph"])
    # print(f"The average hyperedge contains {avg_size} nodes")

    print(
        f"Loaded {args.data}/{args.dataset} | X={X.shape} | Y={Y.shape} "
        f"| num_classes={G['num_classes']} | split={args.split}"
    )

    return X, Y, G


def get_split_m3_compatible(
    Y: torch.Tensor, val_ratio: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """M3-compatible version of get_split without torch_sparse dependency."""
    indices = torch.arange(len(Y))
    perm = torch.randperm(len(indices))
    val_size = int(val_ratio * len(indices))

    val_idx = indices[perm[:val_size]]
    test_idx = indices[perm[val_size:]]

    return val_idx, test_idx


# Simple config parser for M3 compatibility
class SimpleArgs:
    """Simple args class for M3 compatibility."""

    def __init__(self, config: HGNNConfig):
        self.data = "cocitation"
        self.dataset = "cora"
        self.gpu = config.gpu_id
        self.split = 1
        self.epochs = config.epochs
        self.patience = config.patience
        self.n_runs = config.n_runs
        self.add_encodings = False
        self.encodings = None
        self.normalize_features = config.normalize_features
        self.normalize_encodings = config.normalize_encodings


def create_hypergraph_incidence_matrix(
    G: Dict[str, Any], num_nodes: int
) -> torch.Tensor:
    """
    Create hypergraph incidence matrix from graph format.
    FIXED: Properly handle non-consecutive hyperedge IDs.
    """
    if "hypergraph" in G:
        hyperedges = G["hypergraph"]
    else:
        # Create default hyperedges
        hyperedges = {i: [i] for i in range(num_nodes)}

    # Remap hyperedge IDs to consecutive integers
    hyperedge_list = list(hyperedges.items())
    num_edges = len(hyperedge_list)

    H = torch.zeros(num_nodes, num_edges)

    # Use consecutive edge indices
    for new_edge_idx, (old_edge_id, nodes) in enumerate(hyperedge_list):
        for node in nodes:
            if node < num_nodes:
                H[node, new_edge_idx] = 1.0

    # Add self-loops for better connectivity
    self_loops = torch.eye(num_nodes)
    H = torch.cat([H, self_loops], dim=1)

    return H


def train_single_run(
    X: torch.Tensor,
    Y: torch.Tensor,
    G: Dict[str, Any],
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    config: HGNNConfig,
    device: torch.device,
    wandb_run=None,  # Add this parameter
    run_id: int = 0,  # Add run identifier
    global_step: int = 0,  # Add global step counter
) -> Tuple[float, float, float]:
    """
    Train HGNN for a single run.

    Args:
        wandb_run: Optional W&B run for logging during training
        run_id: Identifier for this run (for W&B logging)
        global_step: Global step counter for W&B logging
    """
    # Move to device
    X = X.to(device)
    Y = Y.to(device)

    # Create hypergraph incidence matrix
    H = create_hypergraph_incidence_matrix(G, X.shape[0])
    H = H.to(device)

    # Create model
    num_classes = G["num_classes"]
    model = HGNN(H, X.shape[1], num_classes, config)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Training loop
    best_val_acc = 0.0
    best_test_acc = 0.0
    bad_counter = 0
    patience = config.patience
    epochs = config.epochs

    # Define a per-run step so steps can restart from 0 for each run
    if wandb_run is not None and WANDB_AVAILABLE:
        wandb.define_metric(f"run_{run_id}/step")
        wandb.define_metric(f"run_{run_id}/*", step_metric=f"run_{run_id}/step")

    # Create progress bar
    pbar = tqdm(range(epochs), desc="Training", leave=False)

    for epoch in pbar:
        # Train
        model.train()
        optimizer.zero_grad()
        Z = model(X)
        loss = F.nll_loss(Z[train_idx], Y[train_idx])
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            Z = model(X)

            train_acc = accuracy(Z[train_idx], Y[train_idx])
            val_acc = accuracy(Z[val_idx], Y[val_idx])
            test_acc = accuracy(Z[test_idx], Y[test_idx])

            # Track best validation accuracy (UNIGNN style)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc  # Test acc when val was best
                bad_counter = 0
            else:
                bad_counter += 1
                if bad_counter >= patience:
                    break

            # Log to W&B during training
            if wandb_run is not None and WANDB_AVAILABLE:
                try:
                    wandb.log(
                        {
                            f"run_{run_id}/step": epoch,
                            f"run_{run_id}/train_loss": loss.item(),
                            f"run_{run_id}/train_acc": train_acc,
                            f"run_{run_id}/val_acc": val_acc,
                            f"run_{run_id}/test_acc": test_acc,
                            f"run_{run_id}/best_val_acc": best_val_acc,
                            f"run_{run_id}/best_test_acc": best_test_acc,
                            f"run_{run_id}/learning_rate": config.learning_rate,
                        }
                    )
                except Exception:
                    pass

            # Update progress bar with accuracies
            pbar.set_postfix(
                {
                    "Train": f"{train_acc:.2f}%",
                    "Val": f"{val_acc:.2f}%",
                    "Test": f"{test_acc:.2f}%",
                    "Best_Val": f"{best_val_acc:.2f}%",
                    "Best_Test": f"{best_test_acc:.2f}%",
                }
            )

    final_test_acc = test_acc
    # Always print the key results, regardless of verbose setting
    print(f"Final test accuracy: {final_test_acc:.2f}%")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Clean up
    del model, optimizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return best_val_acc, best_test_acc, final_test_acc


def setup_wandb(
    data_type: str, dataset_name: str, encoding_type: str, config: HGNNConfig
):
    """Initialize W&B run with proper configuration."""
    if not WANDB_AVAILABLE:
        print("W&B not available - skipping logging")
        return None

    try:
        # Set environment variables if not already set
        if not os.environ.get("WANDB_ENTITY"):
            os.environ["WANDB_ENTITY"] = "weber-geoml-harvard-university"
        if not os.environ.get("WANDB_PROJECT"):
            os.environ["WANDB_PROJECT"] = "hgnn-experiments"

        # Login if not already logged in
        if not wandb.run:
            wandb.login()

        # Create a unique run name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{data_type}_{dataset_name}_{encoding_type}_{timestamp}"

        # Initialize run with project and config
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "hgnn-experiments"),
            entity=os.environ.get("WANDB_ENTITY", "weber-geoml-harvard-university"),
            config={
                "data_type": data_type,
                "dataset_name": dataset_name,
                "encoding_type": encoding_type,
                "hidden_dims": config.hidden_dims,
                "dropout_rate": config.dropout_rate,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "epochs": config.epochs,
                "patience": config.patience,
                "val_ratio": config.val_ratio,
                "n_runs": config.n_runs,
                "n_seeds": config.n_seeds,
                "runs_per_seed": config.runs_per_seed,
                "normalize_features": config.normalize_features,
                "normalize_encodings": config.normalize_encodings,
                "timestamp": timestamp,
            },
            name=run_name,
            tags=[data_type, dataset_name, encoding_type, "hgnn"],
            notes=f"HGNN experiment on {data_type}/{dataset_name} with {encoding_type} encoding",
        )
        print(f"  ✓ W&B run initialized: {run.name}")
        print(f"  ✓ W&B run URL: {run.get_url()}")
        return run
    except Exception as e:
        print(f"  ✗ W&B initialization failed: {e}")
        traceback.print_exc()
        return None


def run_experiments_for_encoding(
    data_type: str,
    dataset_name: str,
    encoding_type: str,
    config: HGNNConfig,
    device: torch.device,
    n_runs: int = 80,
    use_best_params: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run experiments for a specific encoding type.

    Returns:
        Dictionary with results including accuracies and hyperparameters
    """
    print(f"\n--- Encoding: {encoding_type} ---")
    print(f"Config: {config.to_dict()}")

    # Initialize W&B run
    wandb_run = setup_wandb(data_type, dataset_name, encoding_type, config)

    # Results storage
    all_best_test_accs = []
    all_final_test_accs = []
    all_best_val_accs = []

    # Check for pre-computed encodings
    precomputed_data = load_precomputed_encoding(
        encoding_type, f"{data_type}_{dataset_name}"
    )

    # Always start with a default result so return is safe
    result = {
        "dataset": f"{data_type}_{dataset_name}",
        "encoding": encoding_type,
        "config": config.to_dict(),
        "mean_test_acc_best_val": 0.0,
        "std_test_acc_best_val": 0.0,
        "mean_final_test_acc": 0.0,
        "std_final_test_acc": 0.0,
        "mean_best_val_acc": 0.0,
        "std_best_val_acc": 0.0,
        "num_runs": 0,
        "all_best_test_accs": [],
        "all_final_test_accs": [],
        "precomputed_available": precomputed_data is not None,
    }

    # If encoding requires a file and it’s missing, skip cleanly with debug info
    if encoding_type != "none" and precomputed_data is None:
        print(f"  [ENC DEBUG] Skipping {encoding_type}: precomputed not available")
        if WANDB_AVAILABLE and wandb_run is not None:
            try:
                wandb.log(
                    {
                        "data/dataset": f"{data_type}_{dataset_name}",
                        "data/encoding": encoding_type,
                        "data/precomputed_available": False,
                        "runs/successful": 0,
                        "runs/total": config.n_runs,
                        "runs/success_rate": 0.0,
                        "error": "No precomputed encodings; skipped",
                    }
                )
            except Exception:
                pass
            wandb_run.finish()
        return result

    # Get best hyperparameters if requested
    if use_best_params:
        best_params = get_best_hyperparameters(data_type, dataset_name, encoding_type)
        print(
            f"Using best hyperparameters: accuracy={best_params.accuracy:.4f} ± {best_params.std:.4f}"
        )

        # Override default parameters with best ones
        kwargs.update(
            {
                "hidden_dims": best_params.hidden_dims,
                "dropout_rate": best_params.dropout_rate,
                "learning_rate": best_params.learning_rate,
                "weight_decay": best_params.weight_decay,
                "epochs": best_params.epochs,
                "patience": best_params.patience,
                "val_ratio": best_params.val_ratio,
                "normalize_features": best_params.normalize_features,
                "normalize_encodings": best_params.normalize_encodings,
            }
        )

    # Run experiments based on config
    global_step = 0  # Initialize global step counter

    for seed in range(2, 2 + config.n_seeds):  # Seeds 2-9 (8 seeds)
        print(f"  Seed {seed}:", end=" ")

        set_seed(seed)

        for run in range(1, 1 + config.runs_per_seed):  # 10 runs per seed
            try:
                # Create args for this run
                args = SimpleArgs(config)
                args.data = data_type
                args.dataset = dataset_name
                args.split = run

                # Load base data
                X_base, Y, G_base = load_base_data(args)

                # Apply encoding if needed
                if encoding_type == "none":
                    X = X_base.clone()
                    G = G_base.copy()
                elif precomputed_data is not None:
                    # Use pre-computed encoding
                    X = torch.tensor(precomputed_data["features"], dtype=torch.float32)
                    G = G_base.copy()  # Keep same hypergraph structure
                    G["num_features"] = X.shape[1]
                else:
                    # Skip if no pre-computed encoding available
                    print("X", end="")
                    continue

                # Validate data after encoding
                assert Y.dim() == 1, f"Y must be 1D, got {Y.shape}"
                assert (
                    X.shape[0] == Y.shape[0]
                ), f"X and Y size mismatch: {X.shape[0]} vs {Y.shape[0]}"

                # Get data splits for this run
                _, train_idx, test_idx = load(args)
                val_idx, test_idx = get_split_m3_compatible(
                    Y[test_idx], config.val_ratio
                )

                # Convert to tensors and move to device
                train_idx = torch.LongTensor(train_idx).to(device)
                val_idx = torch.LongTensor(val_idx).to(device)
                test_idx = torch.LongTensor(test_idx).to(device)

                # Train single run
                best_val_acc, best_test_acc, final_test_acc = train_single_run(
                    X,
                    Y,
                    G,
                    train_idx,
                    val_idx,
                    test_idx,
                    config,
                    device,
                    wandb_run=wandb_run,  # Pass W&B run
                    run_id=len(all_best_test_accs),  # Use current run count as ID
                    global_step=global_step,  # Use global step counter
                )

                # Store results
                all_best_val_accs.append(best_val_acc)
                all_best_test_accs.append(best_test_acc)  # KEY METRIC (UniGNN style)
                all_final_test_accs.append(final_test_acc)

                # Print progress
                if run % 2 == 0:
                    print(f"{run}", end="")
                else:
                    print(".", end="")

            except Exception:
                print("E", end="")  # Error marker
                if config.verbose:
                    traceback.print_exc()
                continue

        print()  # New line after each seed

    # Calculate statistics
    if len(all_best_test_accs) > 0:
        # Primary metric: test accuracy when validation was best (UniGNN style)
        mean_best_test = np.mean(all_best_test_accs)
        std_best_test = np.std(all_best_test_accs)

        # Additional metrics
        mean_final_test = np.mean(all_final_test_accs)
        std_final_test = np.std(all_final_test_accs)

        mean_best_val = np.mean(all_best_val_accs)
        std_best_val = np.std(all_best_val_accs)

        result = {
            "dataset": f"{data_type}_{dataset_name}",
            "encoding": encoding_type,
            "config": config.to_dict(),
            "mean_test_acc_best_val": mean_best_test,  # PRIMARY METRIC
            "std_test_acc_best_val": std_best_test,
            "mean_final_test_acc": mean_final_test,
            "std_final_test_acc": std_final_test,
            "mean_best_val_acc": mean_best_val,
            "std_best_val_acc": std_best_val,
            "num_runs": len(all_best_test_accs),
            "all_best_test_accs": all_best_test_accs,
            "all_final_test_accs": all_final_test_accs,
            "precomputed_available": precomputed_data is not None,
        }

        print(
            f"  ✓ {encoding_type}: {mean_best_test:.4f} ± {std_best_test:.4f} ({len(all_best_test_accs)}/{config.n_runs} runs)"
        )

        # Log to wandb if available
        if WANDB_AVAILABLE and wandb_run is not None:
            try:
                # Log comprehensive results
                wandb.log(
                    {
                        # Primary metrics
                        "test/mean_acc_best_val": mean_best_test,  # PRIMARY METRIC
                        "test/std_acc_best_val": std_best_test,
                        "test/mean_acc_final": mean_final_test,
                        "test/std_acc_final": std_final_test,
                        "val/mean_acc_best": mean_best_val,
                        "val/std_acc_best": std_best_val,
                        # Hyperparameters
                        "hyperparams/learning_rate": config.learning_rate,
                        "hyperparams/hidden_dims": config.hidden_dims,
                        "hyperparams/dropout_rate": config.dropout_rate,
                        "hyperparams/weight_decay": config.weight_decay,
                        "hyperparams/epochs": config.epochs,
                        "hyperparams/patience": config.patience,
                        "hyperparams/val_ratio": config.val_ratio,
                        # Dataset and encoding info
                        "data/dataset": f"{data_type}_{dataset_name}",
                        "data/encoding": encoding_type,
                        "data/precomputed_available": precomputed_data is not None,
                        # Run statistics
                        "runs/successful": len(all_best_test_accs),
                        "runs/total": config.n_runs,
                        "runs/success_rate": len(all_best_test_accs) / config.n_runs,
                        # Additional metrics
                        "metrics/best_test_acc": (
                            max(all_best_test_accs) if all_best_test_accs else 0
                        ),
                        "metrics/worst_test_acc": (
                            min(all_best_test_accs) if all_best_test_accs else 0
                        ),
                        "metrics/median_test_acc": (
                            np.median(all_best_test_accs) if all_best_test_accs else 0
                        ),
                        # Individual run results (for detailed analysis)
                        "runs/all_best_test_accs": all_best_test_accs,
                        "runs/all_final_test_accs": all_final_test_accs,
                        "runs/all_best_val_accs": all_best_val_accs,
                    }
                )
                print(f"  ✓ Logged results to W&B: {wandb_run.get_url()}")
            except Exception as e:
                print(f"  ✗ W&B logging failed: {e}")
                traceback.print_exc()

    # Clean up W&B run
    if wandb_run is not None:
        wandb_run.finish()

    return result


@dataclass
class CLIOptions:
    """CLI options to control what runs."""

    data: Optional[str] = None  # 'coauthorship' or 'cocitation'
    dataset: Optional[str] = None  # e.g., 'cora', 'dblp', 'citeseer', 'pubmed'
    encoding: Optional[str] = None  # one of ENCODING_TYPES
    n_runs: Optional[int] = None  # total runs per (dataset, encoding)
    use_best_params: bool = False
    wandb_enabled: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None


def parse_cli_options() -> CLIOptions:
    """Parse command-line arguments and return structured options."""
    parser = argparse.ArgumentParser(description="HGNN UniGNN-Compatible Runner")
    parser.add_argument("--data", choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--dataset")
    parser.add_argument("--encoding", choices=ENCODING_TYPES)
    parser.add_argument("--n_runs", type=int)
    parser.add_argument("--use_best_params", action="store_true")
    parser.add_argument("--wandb_enabled", action="store_true")
    parser.add_argument("--wandb_project")
    parser.add_argument("--wandb_entity")

    args = parser.parse_args()
    return CLIOptions(
        data=args.data,
        dataset=args.dataset,
        encoding=args.encoding,
        n_runs=args.n_runs,
        use_best_params=args.use_best_params,
        wandb_enabled=args.wandb_enabled,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )


def main() -> None:
    """Main function to run all experiments with configurable hyperparameters."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("=" * 80)
    print("HGNN UniGNN-Compatible Experiments (Configurable)")
    print("=" * 80)

    # Parse CLI
    opts = parse_cli_options()

    # Configure W&B env from CLI
    if not opts.wandb_enabled:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        if opts.wandb_project:
            os.environ["WANDB_PROJECT"] = opts.wandb_project
        if opts.wandb_entity:
            os.environ["WANDB_ENTITY"] = opts.wandb_entity

    # Base configuration
    config = get_default_config()

    # If user specifies total runs, collapse to single seed and set runs_per_seed
    if opts.n_runs is not None:
        config.n_runs = opts.n_runs
        config.n_seeds = 1
        config.runs_per_seed = opts.n_runs

    # Optionally use dataset-specific configs (by dataset name)
    dataset_configs = get_dataset_specific_configs()

    # Select datasets from CLI
    datasets = DATASET_CONFIGS
    if opts.data is not None:
        if opts.dataset is not None:
            datasets = {opts.data: [opts.dataset]}
        else:
            datasets = {opts.data: datasets[opts.data]}

    # Select encodings from CLI
    encoding_types = [opts.encoding] if opts.encoding else ENCODING_TYPES

    # Print plan: datasets and encodings
    total_datasets = sum(len(v) for v in datasets.values())
    print(f"Testing {len(encoding_types)} encodings on {total_datasets} datasets")
    print(f"Configuration: {config.to_dict()}")

    print("Datasets to run:")
    for data_type, names in datasets.items():
        print(f"  - {data_type}: {', '.join(names)}")
    print(f"Encodings to run: {', '.join(encoding_types)}")
    print(
        f"Runs per (dataset, encoding): {config.n_runs}  "
        f"(n_seeds={config.n_seeds}, runs_per_seed={config.runs_per_seed})"
    )

    # Run experiments
    all_results = []

    for data_type, dataset_names in datasets.items():
        for dataset_name in dataset_names:
            print(f"\n{'='*60}")
            print(f"Dataset: {data_type}/{dataset_name}")
            print(f"{'='*60}")

            # Use dataset-specific config if available
            current_config = dataset_configs.get(dataset_name, config)

            for encoding_type in encoding_types:
                try:
                    result = run_experiments_for_encoding(
                        data_type=data_type,
                        dataset_name=dataset_name,
                        encoding_type=encoding_type,
                        config=current_config,
                        device=device,
                        n_runs=current_config.n_runs,
                        use_best_params=opts.use_best_params,
                    )
                    all_results.append(result)

                except Exception as e:
                    print(f"  ✗ {encoding_type} failed completely: {e}")
                    if config.verbose:
                        traceback.print_exc()
                    continue

    # Create summary table
    print("\n" + "=" * 100)
    print("FINAL RESULTS SUMMARY")
    print("=" * 100)

    df = pd.DataFrame(all_results)

    # Display table
    print(
        f"{'Dataset':<20} {'Encoding':<20} {'Test Acc (Best Val)':<20} {'Std':<10} {'Runs':<8} {'Pre-comp':<10}"
    )
    print("-" * 100)

    for _, row in df.iterrows():
        dataset = row["dataset"]
        encoding = row["encoding"]
        mean_acc = row["mean_test_acc_best_val"]
        std_acc = row["std_test_acc_best_val"]
        num_runs = row["num_runs"]
        precomputed = "Yes" if row["precomputed_available"] else "No"

        if num_runs > 0:
            acc_str = f"{mean_acc:.4f} ± {std_acc:.4f}"
        else:
            acc_str = "FAILED"

        print(
            f"{dataset:<20} {encoding:<20} {acc_str:<20} {std_acc:<10.4f} {num_runs:<8} {precomputed:<10}"
        )

    # Save detailed results
    if config.save_results:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"hgnn_results_{timestamp}.csv"
        df.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")

    # UniGNN-style statistical summary
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY (UniGNN Style)")
    print("=" * 80)

    successful_results = df[df["num_runs"] > 0]
    if len(successful_results) > 0:
        for _, row in successful_results.iterrows():
            encoding = row["encoding"]
            dataset = row["dataset"]

            best_test_mean = row["mean_test_acc_best_val"]
            best_test_std = row["std_test_acc_best_val"]
            final_test_mean = row["mean_final_test_acc"]
            final_test_std = row["std_final_test_acc"]

            print(f"\n{dataset} - {encoding}:")
            print(
                f"  Average test accuracy for best val: {best_test_mean:.4f} ± {best_test_std:.4f}"
            )
            print(
                f"  Average final test accuracy: {final_test_mean:.4f} ± {final_test_std:.4f}"
            )

        # Find best performing encoding
        best_idx = successful_results["mean_test_acc_best_val"].idxmax()
        best_result = successful_results.loc[best_idx]

        print("\n" + "=" * 60)
        print("BEST PERFORMING ENCODING")
        print("=" * 60)
        print(f"Dataset: {best_result['dataset']}")
        print(f"Encoding: {best_result['encoding']}")
        print(
            f"Test accuracy (best val): {best_result['mean_test_acc_best_val']:.4f} ± {best_result['std_test_acc_best_val']:.4f}"
        )
        print(f"Successful runs: {best_result['num_runs']}/80")

    print("\nAll experiments completed!")
    print(f"Total runtime: {time.time() - time.time():.2f} seconds")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
