""" Equivalent of train_val.py, but for hypergraph classification"""

import datetime
import os
import pickle
import shutil
import time
from random import sample

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

# File originally taken from UniGCN repo

# Check if CUDA is available and move tensors to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_accs: list[float] = []
best_val_accs: list[float] = []
best_test_accs: list[float] = []


args = config.parse()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

use_norm: str = "use-norm" if args.use_norm else "no-norm"
add_self_loop: str = "add-self-loop" if args.add_self_loop else "no-self-loop"

model_name: str = args.model_name
nlayer: int = args.nlayer
dirname = f"{datetime.datetime.now()}".replace(" ", "_").replace(":", ".")

data_split: list[float] = [0.5, 0.25, 0.25]


# load the dataset specified in args
dataset_name = args.dataset_hypergraph_classification
dataset_path = (
    f"data/hypergraph_classification_datasets/{dataset_name}_hypergraphs.pickle"
)

print("\n=== Dataset Information ===")
print(f"Loading dataset: {dataset_name}")
print(f"From path: {dataset_path}")

with open(dataset_path, "rb") as f:
    current_dataset = pickle.load(f)

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


# pre-compute degEs, degVs, degE2s?
# for dataset_name, dataset_dict in datasets.items():
#     print(f"Processing dataset: {dataset_name}")

#     for hg_dict in dataset_dict:
#         hypergraph = hg_dict.get("hypergraph")
#         features = hg_dict.get("features")
#         calculate_V_E(features, hypergraph)


# split the data
# TODO: to fix
def split_data(data, split) -> tuple:
    """Splits the data into train, val, test sets

    Here the data are list of hypergraphs

    Args:
        data:
            one of the list of hypergraphs
        split:
            TODO
    Returns:
        the train indices
        the val indices
        the test indices

        correspondin to a classification of each hypergraph in the least to be in the
        train, test or val.


    """
    n: int = len(data)
    indices = np.random.permutation(n)
    train_idx = indices[: int(split[0] * n)]
    val_idx = indices[int(split[0] * n) : int((split[0] + split[1]) * n)]
    test_idx = indices[int((split[0] + split[1]) * n) :]
    return data[train_idx], data[val_idx], data[test_idx]


# TODO: or get inspired from previous function
# now
# Might have to do something more along these lines:
def get_split(Y, p: float = 0.2) -> tuple[list[int], list[int]]:
    """Splits Y into a test and val set.

    Args:
        Y:
            the labels of nodes.
        p:
            the proportion of nodes in the val set

    Returns:
        val_idx:
            the indices of nodes in the val set
        test_idx:
            the indices of nodes in the test set

    """
    nclass: int = len(torch.unique(Y))  # number of different labels
    Y: list = Y.tolist()
    N: int = len(Y)  # number of nodes
    D: list = [[] for _ in range(nclass)]
    for i, y in enumerate(Y):
        # print(f"i is {i} and y is {y}")
        D[y].append(i)
    k: int = int(N * p / nclass)
    val_idx: list[int] = torch.cat(
        [torch.LongTensor(sample(idxs, k)) for idxs in D]
    ).tolist()
    test_idx: list[int] = list(set(range(N)) - set(val_idx))

    return val_idx, test_idx


args = config.parse()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

use_norm: str = "use-norm" if args.use_norm else "no-norm"
add_self_loop: str = "add-self-loop" if args.add_self_loop else "no-self-loop"

# dataname: str = f"{args.data}_{args.dataset}" TODO: to clean up for ability in condig/args

# model name (eg UniCGN)
model_name: str = args.model_name
# depth of the neural networks
nlayer: int = args.nlayer
dirname = f"{datetime.datetime.now()}".replace(" ", "_").replace(":", ".")


# darta already loaded. Assume the features + encodings are already present on the loaded file
X: torch.Tensor  # the features
Y: torch.Tensor  # the labels
G: dict  # the whole hypergraph
features_shape = current_dataset[0]["features"].shape


out_dir: str = path.Path(f"./{args.out_dir}/{model_name}_{nlayer}_hg_classification/")
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.makedirs_p()
baselogger = get_logger("base logger", f"{out_dir}/logging.log", not args.nostdout)
resultlogger = get_logger("result logger", f"{out_dir}/result.log", not args.nostdout)
baselogger.info(args)

resultlogger.info(args)

# TODO: direty right now. TO fix
# the labels
Y: list[int]
Y = []
for hg in current_dataset:
    # Convert numpy array label to integer
    label = hg["labels"]
    if isinstance(label, np.ndarray):
        label = label.item()  # Convert single-element array to scalar
    Y.append(label)

# Convert to tensor
Y = torch.tensor(Y, dtype=torch.long)
Y = Y.view(-1)  # Ensure it's 1D

print("\nLabel Information:")
print(f"Number of labels: {len(Y)}")
print(f"Label type: {Y.dtype}")
print(f"Unique labels: {torch.unique(Y).tolist()}")

# Keep track of the test accuracy for the best val accuracy
overall_test_accuracies_best_val: list[float] = []

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
        # TODO: to clean up here
        num_hypergraphs = len(current_dataset)
        indices = list(range(num_hypergraphs))
        # TODO: make sure that the radom state does not give a 'trivial' split
        # as the first 500 hgs are labels 0 and the last 500 are label 1.
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

        print(train_idx)
        print(test_idx)
        val_idx: list[int]
        test_idx: list[int]
        Y = torch.tensor(Y)
        Y = Y.view(-1)
        # we split the test set because we have train, test, val
        val_idx, test_idx = get_split(Y[test_idx], 0.2)
        train_idx: torch.Tensor = torch.LongTensor(train_idx).to(device)
        val_idx: torch.Tensor = torch.LongTensor(val_idx).to(device)
        test_idx: torch.Tensor = torch.LongTensor(test_idx).to(device)

        print(f"\nSplit Statistics:")
        print(f"Training set size: {len(train_idx)} hypergraphs")
        print(f"Validation set size: {len(val_idx)} hypergraphs")
        print(f"Test set size: {len(test_idx)} hypergraphs")

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
        for epoch in tqdm(range(args.epochs), desc=f"Epochs (Run {run})"):
            # train
            tic_epoch = time.time()
            model.train()

            optimizer.zero_grad()
            Z = model.forward_hypergraph_classification(
                current_dataset
            )  # this call forward.
            Z = torch.stack(Z).squeeze(1)
            train_idx = train_idx.long()  # Convert to integer tensor
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
