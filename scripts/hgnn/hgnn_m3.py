# -*- coding: utf-8 -*-
"""UniGNN-Compatible HGNN with 80 runs and pre-computed encodings"""

import datetime
import os
import sys
import time
import pickle
from pathlib import Path
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm

# Add necessary paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "unignn"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Import only the modules that don't depend on torch_sparse
from encodings_hnns.data_handling import load

warnings.filterwarnings("ignore")
os.environ["TORCH"] = torch.__version__


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

    filename = filename_map[encoding_type]

    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f:
                dataset = pickle.load(f)
            print(f"  Loaded pre-computed encoding from {filename}")
            return dataset
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
            import traceback

            traceback.print_exc()
            return None
    else:
        print(f"  Pre-computed encoding not found: {filename}")
        return None


def load_base_data(args) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Load base data using UniGNN's data loading function.
    FIXED: Handle tuple return from load() function.
    """
    # Load data using UniGNN's function - returns (dataset, train, test)
    dataset, train_idx, test_idx = load(args)

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
        print(f"  Converting 2D labels {Y.shape} to 1D")
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

    print(f"number of hyperedges is {len(G['hypergraph'])}")
    print(
        f"The hypergraph {args.data}/{args.dataset} has {len(G['hypergraph'])} hyperedges where authors are hyperedges"
    )

    # Calculate average hyperedge size
    total_nodes_in_edges = sum(len(nodes) for nodes in G["hypergraph"].values())
    avg_size = total_nodes_in_edges / len(G["hypergraph"])
    print(f"The average hyperedge contains {avg_size} nodes")

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

    def __init__(self):
        self.data = "cocitation"
        self.dataset = "cora"
        self.gpu = 0
        self.split = 1
        self.epochs = 500
        self.patience = 50
        self.n_runs = 10  # 10 runs per seed
        self.add_encodings = False
        self.encodings = None
        self.normalize_features = False
        self.normalize_encodings = False


class HGNN(nn.Module):
    def __init__(
        self, H: torch.Tensor, in_size: int, out_size: int, hidden_dims: int = 16
    ):
        """Hypergraph Neural Network model compatible with M3."""
        super().__init__()

        self.W1 = nn.Linear(in_size, hidden_dims)
        self.W2 = nn.Linear(hidden_dims, out_size)
        self.dropout = nn.Dropout(0.5)

        # Convert to dense for M3 compatibility
        H_dense = H.to_dense() if H.is_sparse else H

        # Handle edge cases
        if H_dense.numel() == 0 or H_dense.sum() == 0:
            num_nodes = in_size
            H_dense = torch.eye(num_nodes)
            H_dense = torch.cat([H_dense, torch.eye(num_nodes)], dim=1)

        # Compute node degree
        d_V = H_dense.sum(1)
        d_V = torch.where(d_V == 0, torch.ones_like(d_V), d_V)

        # Compute edge degree
        d_E = H_dense.sum(0)
        d_E = torch.where(d_E == 0, torch.ones_like(d_E), d_E)

        # Compute Laplacian matrices
        D_v_invsqrt = torch.diag(d_V**-0.5)
        D_e_inv = torch.diag(d_E**-1)
        n_edges = d_E.shape[0]
        B = torch.eye(n_edges)

        # Compute Laplacian: L = D_v^{-1/2} H B D_e^{-1} H^T D_v^{-1/2}
        self.L = D_v_invsqrt @ H_dense @ B @ D_e_inv @ H_dense.T @ D_v_invsqrt

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass through the HGNN."""
        X = self.L @ self.W1(self.dropout(X))
        X = F.relu(X)
        X = self.L @ self.W2(self.dropout(X))
        return F.log_softmax(X, dim=1)


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
    args: Any,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Train HGNN for a single run.

    Returns:
        best_val_acc: Best validation accuracy achieved
        best_test_acc: Test accuracy when validation was best (KEY METRIC)
        final_test_acc: Final test accuracy
    """
    # Move to device
    X = X.to(device)
    Y = Y.to(device)

    # Create hypergraph incidence matrix
    H = create_hypergraph_incidence_matrix(G, X.shape[0])
    H = H.to(device)

    # Create model
    num_classes = G["num_classes"]
    model = HGNN(H, X.shape[1], num_classes, hidden_dims=16)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Training loop
    best_val_acc = 0.0
    best_test_acc = 0.0
    bad_counter = 0
    patience = args.patience
    epochs = args.epochs

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
    print(f"Final test accuracy: {final_test_acc}")
    print(f"Best test accuracy: {best_test_acc}")
    print(f"Best validation accuracy: {best_val_acc}")

    # Clean up
    del model, optimizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return best_val_acc, best_test_acc, final_test_acc


def run_80_experiments_for_encoding(
    data_type: str, dataset_name: str, encoding_type: str, device: torch.device
) -> Dict[str, Any]:
    """
    Run 80 experiments (8 seeds × 10 runs) for a specific encoding.

    Returns:
        Dictionary with results and statistics
    """
    print(f"\n--- Encoding: {encoding_type} ---")

    # Results storage
    all_best_test_accs = []
    all_final_test_accs = []
    all_best_val_accs = []

    # Check for pre-computed encodings
    precomputed_data = load_precomputed_encoding(
        encoding_type, f"{data_type}_{dataset_name}"
    )

    # 8 seeds × 10 runs = 80 total experiments
    for seed in range(2, 10):  # Seeds 2-9 (8 seeds)
        print(f"  Seed {seed}:", end=" ")

        set_seed(seed)

        for run in range(1, 11):  # 10 runs per seed
            try:
                # Create args for this run
                args = SimpleArgs()
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
                    print(f"X", end="")
                    continue

                # Validate data after encoding
                assert Y.dim() == 1, f"Y must be 1D, got {Y.shape}"
                assert (
                    X.shape[0] == Y.shape[0]
                ), f"X and Y size mismatch: {X.shape[0]} vs {Y.shape[0]}"

                # Get data splits for this run
                _, train_idx, test_idx = load(args)
                val_idx, test_idx = get_split_m3_compatible(Y[test_idx], 0.2)

                # Convert to tensors and move to device
                train_idx = torch.LongTensor(train_idx).to(device)
                val_idx = torch.LongTensor(val_idx).to(device)
                test_idx = torch.LongTensor(test_idx).to(device)

                # Train single run
                best_val_acc, best_test_acc, final_test_acc = train_single_run(
                    X, Y, G, train_idx, val_idx, test_idx, args, device
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

            except Exception as e:
                print(f"E", end="")  # Error marker
                import traceback

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
            f"  ✓ {encoding_type}: {mean_best_test:.4f} ± {std_best_test:.4f} ({len(all_best_test_accs)}/80 runs)"
        )
        return result
    else:
        print(f"  ✗ {encoding_type}: No successful runs")
        return {
            "dataset": f"{data_type}_{dataset_name}",
            "encoding": encoding_type,
            "mean_test_acc_best_val": 0.0,
            "std_test_acc_best_val": 0.0,
            "mean_final_test_acc": 0.0,
            "std_final_test_acc": 0.0,
            "mean_best_val_acc": 0.0,
            "std_best_val_acc": 0.0,
            "num_runs": 0,
            "all_best_test_accs": [],
            "all_final_test_accs": [],
            "precomputed_available": False,
        }


def main():
    """Main function to run all 80-run experiments."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("=" * 80)
    print("HGNN UniGNN-Compatible Experiments (80 runs per encoding)")
    print("=" * 80)

    # Define datasets and encodings to test
    datasets = {
        "cocitation": ["cora"],  # Can expand to ['cora', 'citeseer', 'pubmed']
        # 'coauthorship': ['cora', 'dblp']  # Add more datasets later
    }

    encoding_types = [
        "none",
        "degree",
        # "random_walk_EE",
        # "laplacian_Hodge",
        "curvature_FRC",
    ]

    print(
        f"Testing {len(encoding_types)} encodings on {sum(len(v) for v in datasets.values())} datasets"
    )
    print("Each encoding: 8 seeds × 10 runs = 80 experiments")
    print("Legend: . = run completed, X = run failed, E = error")

    # Run experiments
    all_results = []

    for data_type, dataset_names in datasets.items():
        for dataset_name in dataset_names:
            print(f"\n{'='*60}")
            print(f"Dataset: {data_type}/{dataset_name}")
            print(f"{'='*60}")

            for encoding_type in encoding_types:
                try:
                    result = run_80_experiments_for_encoding(
                        data_type, dataset_name, encoding_type, device
                    )
                    all_results.append(result)

                except Exception as e:
                    print(f"  ✗ {encoding_type} failed completely: {e}")
                    import traceback

                    traceback.print_exc()
                    continue

    # Create summary table
    print("\n" + "=" * 100)
    print("FINAL RESULTS SUMMARY (UniGNN Style - 80 runs per encoding)")
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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"hgnn_unignn_80runs_results_{timestamp}.csv"
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

        print(f"\n" + "=" * 60)
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
