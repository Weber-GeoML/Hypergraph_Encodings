""" Equivalent of train_val.py, but for hypergraph classification"""

import datetime
import os
from random import sample
import pickle
import shutil
import time
import matplotlib.pyplot as plt

import config
import numpy as np
import path
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim import optimizer
from tqdm import tqdm  # Add this import

### configure logger
### configure logger
from uniGCN.logger import get_logger
from uniGCN.prepare import accuracy
from uniGCN.prepare_hg import initialise_for_hypergraph_classification
from split_data_for_hypergraph_classification import get_split

# Print command line arguments
print("\nCommand Line Arguments:")
print("=" * 80)
print(f"Script name: {sys.argv[0]}")
print(f"Arguments passed: {sys.argv[1:]}")
print("=" * 80)

# Print config.py contents
print("\nContents of config.py:")
print("=" * 80)
try:
    with open("config.py", "r") as f:
        print(f.read())
except Exception as e:
    print(f"Error reading config.py: {e}")
print("=" * 80)

# Print default arguments and any overrides from command line
print("\nDefault arguments from config.parse():")
print("=" * 80)
try:
    default_args = config.parse([])  # Parse with empty args list to get defaults
    print("Default values:")
    for arg, value in vars(default_args).items():
        print(f"{arg}: {value}")

    print("\nParsed command line arguments:")
    actual_args = config.parse()  # Parse with actual command line args
    for arg, value in vars(actual_args).items():
        if getattr(default_args, arg) != value:
            print(
                f"{arg}: {value} (overridden from default: {getattr(default_args, arg)})"
            )
except Exception as e:
    print(f"Error getting arguments: {e}")
print("=" * 80)


# At the end of the file, create summary plots
def create_summary_plots(all_results, out_dir):
    """Create summary plots from all runs."""
    plt.figure(figsize=(15, 10))

    # Plot 1: Box plot of final accuracies
    plt.subplot(2, 2, 1)
    final_trains = [
        params["final_train_acc"] for params in all_results["params"].values()
    ]
    final_vals = [params["final_val_acc"] for params in all_results["params"].values()]
    final_tests = [
        params["final_test_acc"] for params in all_results["params"].values()
    ]

    plt.boxplot(
        [final_trains, final_vals, final_tests], labels=["Train", "Validation", "Test"]
    )
    plt.title("Distribution of Final Accuracies")
    plt.ylabel("Accuracy (%)")

    # Plot 2: Learning curves with confidence intervals
    plt.subplot(2, 2, 2)
    max_epochs = max(len(accs) for accs in all_results["train_accs"].values())
    epochs = range(1, max_epochs + 1)

    # Calculate mean and std for each metric
    def get_stats(accs_dict):
        padded_accs = [
            accs + [accs[-1]] * (max_epochs - len(accs)) for accs in accs_dict.values()
        ]
        return np.mean(padded_accs, axis=0), np.std(padded_accs, axis=0)

    train_mean, train_std = get_stats(all_results["train_accs"])
    val_mean, val_std = get_stats(all_results["val_accs"])
    test_mean, test_std = get_stats(all_results["test_accs"])

    plt.plot(epochs, train_mean, "b-", label="Train")
    plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.plot(epochs, val_mean, "g-", label="Validation")
    plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2)
    plt.plot(epochs, test_mean, "r-", label="Test")
    plt.fill_between(epochs, test_mean - test_std, test_mean + test_std, alpha=0.2)

    plt.title("Average Learning Curves (with std)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    # Save results and plot
    results_dir = out_dir / "results"
    results_dir.makedirs_p()

    # Save numerical results
    with open(results_dir / "all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    # Save plot
    plt.tight_layout()
    plt.savefig(results_dir / "summary_plots.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n=== Results Saved ===")
    print(f"All results saved to: {results_dir / 'all_results.pkl'}")
    print(f"Summary plots saved to: {results_dir / 'summary_plots.png'}")


# File originally taken from UniGCN repo

# Check if CUDA is available and move tensors to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Init lists to store accuracy results
test_accs: list[float] = []
best_val_accs: list[float] = []
best_test_accs: list[float] = []
# Keep track of the test accuracy for the best val accuracy
overall_test_accuracies_best_val: list[float] = []

# Parse the arguments from config.py
args = config.parse()

# Set CUDA environment variables
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

use_norm: str = "use-norm" if args.use_norm else "no-norm"
add_self_loop: str = "add-self-loop" if args.add_self_loop else "no-self-loop"
model_name: str = args.model_name
nlayer: int = args.nlayer
dirname = f"{datetime.datetime.now()}".replace(" ", "_").replace(":", ".")

# Define the data split for train, val, test
data_split: list[float] = [0.5, 0.25, 0.25]

out_dir: str = path.Path(f"./{args.out_dir}/{model_name}_{nlayer}_hg_classification/")
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.makedirs_p()

# Configure loggers
baselogger = get_logger("base logger", f"{out_dir}/logging.log", not args.nostdout)
resultlogger = get_logger("result logger", f"{out_dir}/result.log", not args.nostdout)
baselogger.info(args)
resultlogger.info(args)


# Determine which dataset file to load based on config
dataset_name = args.dataset_hypergraph_classification
base_path = "data/hypergraph_classification_datasets"

if args.add_encodings_hg_classification:
    # Construct encoding suffix based on config
    encoding_type = args.encodings.lower()  # e.g., ("RW" "LCP" "Laplacian" "LDP"

    if encoding_type == "rw":
        encoding_suffix = f"with_encodings_{encoding_type}_{args.random_walk_type}"
    elif encoding_type == "lcp":
        encoding_suffix = f"with_encodings_{args.curvature_type}"
    elif encoding_type == "laplacian":
        encoding_suffix = f"with_encodings_lape_{args.laplacian_type}"
    else:  # for ldp
        encoding_suffix = f"with_encodings_{encoding_type}"

    dataset_path = f"{base_path}/{dataset_name}_hypergraphs_{encoding_suffix}.pickle"
else:
    dataset_path = f"{base_path}/{dataset_name}_hypergraphs.pickle"

print("\n=== Dataset Loading ===")
print(f"Dataset name: {dataset_name}")
print(f"Using encodings: {args.add_encodings_hg_classification}")
if args.add_encodings_hg_classification:
    print(f"Encoding type: {args.encodings}")
    if args.encodings.lower() == "rw":
        print(f"Random walk type: {args.random_walk_type}")
    elif args.encodings.lower() == "curvature":
        print(f"Curvature type: {args.curvature_type}")
    elif args.encodings.lower() == "laplacian":
        print(f"Laplacian type: {args.laplacian_type}")
print(f"Loading from: {dataset_path}")

try:
    with open(dataset_path, "rb") as f:
        current_dataset = pickle.load(f)
    print(
        f"Dataset loaded successfully! It contains {len(current_dataset)} hypergraphs"
    )
except FileNotFoundError:
    raise FileNotFoundError(
        f"Dataset file not found: {dataset_path}\n"
        f"Make sure the dataset with the specified encodings exists."
    )

# Add dataset statistics
num_hypergraphs = len(current_dataset)
feature_shape = current_dataset[0]["features"].shape
num_features = feature_shape[1]

# Modified to handle numpy arrays
unique_labels = set()
for hg in current_dataset:
    label = hg["labels"]
    # Convert numpy array to a hashable type (tuple or scalar)
    if isinstance(label, np.ndarray):
        label = tuple(label.flatten())
    unique_labels.add(label)

print(f"\nDataset Statistics:")
print(f"Number of hypergraphs: {num_hypergraphs}")
print(f"Feature dimensions: {feature_shape}")
print(f"Number of features per node: {num_features}")
print(f"Number of unique labels: {len(unique_labels)}")
print(f"Unique labels: {sorted(unique_labels)}")

# Calculate average nodes and edges
avg_nodes = sum(hg["features"].shape[0] for hg in current_dataset) / num_hypergraphs
avg_edges = sum(len(hg["hypergraph"]) for hg in current_dataset) / num_hypergraphs

print(f"\nHypergraph Statistics:")
print(f"Average nodes per hypergraph: {avg_nodes:.2f}")
print(f"Average edges per hypergraph: {avg_edges:.2f}")

# Replace the individual dataset loads with a single dictionary
datasets: dict[str, list[dict]] = {dataset_name: current_dataset}

# TODO later
# so then we don't need to do that in UniGCN/
# pre-compute degEs, degVs, degE2s?
# for dataset_name, dataset_dict in datasets.items():
#     print(f"Processing dataset: {dataset_name}")

#     for hg_dict in dataset_dict:
#         hypergraph = hg_dict.get("hypergraph")
#         features = hg_dict.get("features")
#         calculate_V_E(features, hypergraph)


# darta already loaded. Assume the features + encodings are already present on the loaded file
X: torch.Tensor  # the features
Y: torch.Tensor  # the labels
G: dict  # the whole hypergraph
features_shape = current_dataset[0]["features"].shape


# TODO: direty right now. TO fix
# the labels
Y: list[int]
Y = []
for hg in current_dataset:
    # No need to convert to item() if we want to keep as numpy array
    Y.append(hg["labels"])

# Convert list of numpy arrays to a single tensor
Y = torch.from_numpy(np.stack(Y))  # This preserves the array structure
if len(Y.shape) > 1:
    Y = Y.squeeze()  # Remove extra dimensions
Y = Y.long()  # Convert to long dtype
print(f"Y shape: {Y.shape}")  # Should be 1D now
print(f"Y type: {Y.dtype}")
print(f"Y: \n {Y}")

# When calculating number of classes
nclass = len(np.unique(np.concatenate([hg["labels"] for hg in current_dataset])))
print(f"Number of unique classes: {nclass}")

# _, train_idx, test_idx = load(args)
# TODO: to clean up here
num_hypergraphs = len(current_dataset)
indices = list(range(num_hypergraphs))
# TODO: make sure that the radom state does not give a 'trivial' split
# as the first 500 hgs are labels 0 and the last 500 are label 1.
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=2)


# At the start of the file, after imports
# Create a dictionary to store all accuracies
all_results = {"train_accs": {}, "val_accs": {}, "test_accs": {}, "params": {}}

seed: int
for seed in range(1, 9):
    print(f"The seed is {seed}")
    # gpu, seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    #### configure output directory

    out_dir: str = path.Path(
        f"./{args.out_dir}/{model_name}_{nlayer}_hg_classification/seed_{seed}"
    )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.makedirs_p()

    for run in tqdm(range(1, args.n_runs + 1), desc="Training runs"):
        baselogger.info(f"\n--- Starting run {run}/{args.n_runs} ---")
        run_dir = out_dir / f"{run}"
        run_dir.makedirs_p()

        # load data
        args.split = run
        # _, train_idx, test_idx = load(args)

        # Then split temp into val and test
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.8, random_state=2
        )  # 0.8 of 0.3 = 0.24 for test set

        # After splitting the data
        train_labels = [current_dataset[idx]["labels"] for idx in train_idx]
        val_labels = [current_dataset[idx]["labels"] for idx in val_idx]
        test_labels = [current_dataset[idx]["labels"] for idx in test_idx]

        # Convert numpy arrays to hashable type (tuple or scalar)
        unique_labels = set()
        for label in train_labels + val_labels + test_labels:
            if isinstance(label, np.ndarray):
                label = tuple(label.flatten())
            unique_labels.add(label)
        unique_labels = sorted(unique_labels)

        # Count occurrences of each class
        train_dist = [train_labels.count(label) for label in unique_labels]
        val_dist = [val_labels.count(label) for label in unique_labels]
        test_dist = [test_labels.count(label) for label in unique_labels]

        print("Split Statistics:")
        print(
            f"Training set size: {len(train_idx)} hypergraphs (Class distribution: {train_dist})"
        )
        print(
            f"Validation set size: {len(val_idx)} hypergraphs (Class distribution: {val_dist})"
        )
        print(
            f"Test set size: {len(test_idx)} hypergraphs (Class distribution: {test_dist})"
        )

        # Check for overlap
        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)

        print("\nChecking for overlaps:")
        print(f"Train-Val overlap: {len(train_set & val_set)} indices")
        print(f"Train-Test overlap: {len(train_set & test_set)} indices")
        print(f"Val-Test overlap: {len(val_set & test_set)} indices")

        # model
        (
            model,
            optimizer,
            degVs,
            degEs,
            degE2s,
        ) = initialise_for_hypergraph_classification(current_dataset, args)

        print("\n=== Training Information ===")
        print(f"Model: {model_name}")
        print(f"Number of layers: {nlayer}")
        print(f"Number of runs: {args.n_runs}")
        print(f"Number of epochs: {args.epochs}")
        print(f"Device: {device}")

        baselogger.info(f"Run {run}/{args.n_runs}, Total Epochs: {args.epochs}")
        baselogger.info(model)
        baselogger.info(
            f"total_params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        tic_run = time.time()

        best_val_acc: float = 0
        best_test_acc: float = 0
        test_acc: float = 0
        Z: torch.Tensor | None = None
        bad_counter: int = 0
        test_accs_for_best_val = (
            []
        )  # List to store test accuracy for the best validation accuracy
        train_accs_history = []
        val_accs_history = []
        test_accs_history = []
        for epoch in tqdm(range(args.epochs), desc=f"Epochs (Run {run})"):
            # train
            tic_epoch = time.time()
            model.train()

            optimizer.zero_grad()
            Z = model.forward_hypergraph_classification(
                current_dataset
            )  # this call forward.
            Z = torch.stack(Z).squeeze(1)
            # Convert indices to tensors
            train_idx = torch.tensor(
                train_idx
            ).long()  # Convert list to tensor and then to long dtype
            val_idx = torch.tensor(val_idx).long()
            test_idx = torch.tensor(test_idx).long()
            loss = F.nll_loss(Z[train_idx], Y[train_idx])

            loss.backward()
            optimizer.step()

            train_time = time.time() - tic_epoch

            # eval
            model.eval()
            Z: torch.Tensor = model.forward_hypergraph_classification(
                current_dataset,
            )  # this calls forward
            Z = torch.stack(Z).squeeze(1)
            # gets the trains, test and val accuracy
            train_acc: float = accuracy(Z[train_idx], Y[train_idx])
            test_acc: float = accuracy(Z[test_idx], Y[test_idx])
            val_acc: float = accuracy(Z[val_idx], Y[val_idx])

            # Store accuracies
            train_accs_history.append(train_acc)
            val_accs_history.append(val_acc)
            test_accs_history.append(test_acc)

            # log acc
            if best_val_acc < val_acc:
                best_val_acc: float = val_acc
                best_test_acc: float = test_acc
                bad_counter: int = 0
                test_accs_for_best_val.append(
                    test_acc
                )  # Save the test accuracy when validation accuracy improves
            else:
                bad_counter += 1
                if bad_counter >= args.patience:
                    break
            if epoch % 50 == 0:  # Changed from 20 to 50
                baselogger.info(
                    f"Epoch {epoch:3d} | "
                    f"Loss: {loss:.4f} | "
                    f"Train: {train_acc:.2f}% | "
                    f"Val: {val_acc:.2f}% | "
                    f"Test: {test_acc:.2f}% | "
                    f"Best Test: {best_test_acc:.2f}% | "
                    f"Time: {train_time*1000:.1f}ms"
                )

        resultlogger.info(
            f"Run {run}/{args.n_runs}, best test accuracy: {best_test_acc:.2f}, acc(last): {test_acc:.2f}, total time: {time.time()-tic_run:.2f}s"
        )
        test_accs.append(test_acc)
        best_val_accs.append(best_val_acc)
        best_test_accs.append(best_test_acc)
        overall_test_accuracies_best_val.append(test_accs_for_best_val[-1])

        # Store accuracies with unique key
        run_key = f"seed_{seed}_run_{run}"
        all_results["train_accs"][run_key] = train_accs_history
        all_results["val_accs"][run_key] = val_accs_history
        all_results["test_accs"][run_key] = test_accs_history
        all_results["params"][run_key] = {
            "model": model_name,
            "nlayer": nlayer,
            "dataset": dataset_name,
            "encodings": (
                args.encodings if args.add_encodings_hg_classification else "none"
            ),
            "transformer": args.do_transformer,
            "final_train_acc": train_accs_history[-1],
            "final_val_acc": val_accs_history[-1],
            "final_test_acc": test_accs_history[-1],
            "best_val_acc": best_val_acc,
            "best_test_acc": best_test_acc,
        }

# Call the plotting function at the end
create_summary_plots(all_results, out_dir)

resultlogger.info(f"We had {len(test_accs)} runs")
resultlogger.info(
    f"Average test accuracy for best val: {np.mean(overall_test_accuracies_best_val)} ± {np.std(overall_test_accuracies_best_val)}"
)
resultlogger.info(
    f"Average final test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}"
)
resultlogger.info(
    f"Average best test accuracy: {np.mean(best_test_accs)} ± {np.std(best_test_accs)}"
)

# Add dataset info logging
for dataset_name, dataset in datasets.items():
    baselogger.info(f"\nDataset: {dataset_name}")
    baselogger.info(f"Number of hypergraphs: {len(dataset)}")

    # Calculate average hypergraph size
    avg_nodes = np.mean([hg["features"].shape[0] for hg in dataset])
    avg_edges = np.mean([len(hg["hypergraph"]) for hg in dataset])

    baselogger.info(f"Average nodes per hypergraph: {avg_nodes:.2f}")
    baselogger.info(f"Average edges per hypergraph: {avg_edges:.2f}")

# Improve split logging
baselogger.info(f"\nData split sizes:")
baselogger.info(f"Train set: {len(train_idx)} hypergraphs")
baselogger.info(f"Val set: {len(val_idx)} hypergraphs")
baselogger.info(f"Test set: {len(test_idx)} hypergraphs")

# Add final summary statistics
baselogger.info("\n=== Final Results ===")
baselogger.info(f"Number of completed runs: {len(test_accs)}")
baselogger.info(f"Best validation accuracy: {max(best_val_accs):.2f}%")
baselogger.info(f"Best test accuracy: {max(best_test_accs):.2f}%")

print(f"\nSplit Information:")
print(f"Train set size: {len(train_idx)}")
print(f"Test set size: {len(test_idx)}")

# Check for overlap
train_set = set(train_idx.tolist())
test_set = set(test_idx.tolist())
overlap = train_set.intersection(test_set)

print(f"Overlap between train and test: {len(overlap)} indices")
if len(overlap) > 0:
    print(f"Overlapping indices: {overlap}")

# Debug label shapes
print("\nLabel debug:")
print(f"Original Y shape: {np.stack([hg['labels'] for hg in current_dataset]).shape}")
Y = torch.from_numpy(np.stack([hg["labels"] for hg in current_dataset]))
print(f"Tensor Y shape before squeeze: {Y.shape}")
print(f"Tensor Y shape after squeeze: {Y.squeeze().shape}")
