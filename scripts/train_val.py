""" File taken from UniGCN

This is the file for node level classifications. This is the file
that we call from run_all_general_parallel.sh, run_all_general.sh, run_all_ablation.sh
"""

import datetime
import os
import shutil
import sys
import time
from random import sample

import config
import numpy as np
import path
import torch
import torch.nn.functional as F

# load data
from encodings_hnns.data_handling import load
from uniGCN.calculate_vertex_edges import calculate_V_E

### configure logger
from uniGCN.logger import get_logger
from uniGCN.prepare import accuracy, fetch_data, initialise

# Initialize results dictionary before the training loops
all_results: dict[str, dict[str, list[float] | dict[str, float]]] = {
    "train_accs": {},
    "val_accs": {},
    "test_accs": {},
    "params": {},
}

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
    with open("scripts/config.py", "r") as f:
        print(f.read())
except Exception as e:
    print(f"Error reading config.py: {e}")
print("=" * 80)

# Print default arguments and any overrides from command line
print("=" * 80)
try:
    # Parse command line arguments, starting with defaults
    args = config.parse()

    # Override default values in config with command line arguments
    for arg, value in vars(args).items():
        default_value = getattr(config, arg, None)
        if default_value is not None and value != default_value:
            print(f"Overriding {arg}: {default_value} -> {value}")
            setattr(config, arg, value)  # Update the config attribute

except Exception as e:
    print(f"Error handling arguments: {e}")
    raise e

# Print final configuration
print("\nFinal configuration:")
print("=" * 80)
for arg, value in vars(args).items():
    print(f"{arg}: {value}")
print("=" * 80)


# File originally taken from UniGCN repo

# Check if CUDA is available and move tensors to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_accs: list[float] = []
best_val_accs: list[float] = []
best_test_accs: list[float] = []


# args = config.parse()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

use_norm: str = "use-norm" if args.use_norm else "no-norm"
add_self_loop: str = "add-self-loop" if args.add_self_loop else "no-self-loop"

dataname: str = f"{args.data}_{args.dataset}"
# model name (eg UniCGN)
model_name: str = args.model_name
# depth of the neural networks
nlayer: int = args.nlayer
dirname = f"{datetime.datetime.now()}".replace(" ", "_").replace(":", ".")


# load data
X: torch.Tensor  # the features
Y: torch.Tensor  # the labels
G: dict  # the whole hypergraph
if args.add_encodings:
    print(f"We are adding the {args.encodings} encodings")
else:
    print("We are not adding any encodings")
X, Y, G = fetch_data(
    args,
    add_encodings=args.add_encodings,
    encodings=args.encodings,
    laplacian_type=args.laplacian_type,
    random_walk_type=args.random_walk_type,
    k_rw=args.k_rw,
    curvature_type=args.curvature_type,
    normalize_features=args.normalize_features,  # whether or not to normalize the features
    normalize_encodings=args.normalize_encodings,  # whether or not to normalize the encodings
)
print(f"X are the features \n {X} \n with shape {X.shape}")
print(f"Y are the labels \n {Y}")
print("G is the hg")
V, E, degE, degV, degE2 = calculate_V_E(X, G, args)


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
    Y: list = Y.tolist()
    N: int = len(Y)  # number of nodes
    nclass: int = len(set(Y))  # number of different labels
    D: list = [[] for _ in range(nclass)]
    for i, y in enumerate(Y):
        D[y].append(i)
    k: int = int(N * p / nclass)
    val_idx: list[int] = torch.cat(
        [torch.LongTensor(sample(idxs, k)) for idxs in D]
    ).tolist()
    test_idx: list[int] = list(set(range(N)) - set(val_idx))

    return val_idx, test_idx


# Create a more detailed output directory name
out_dir_components = [
    args.out_dir,
    f"model_{model_name}",
    f"nlayer_{nlayer}",
    f"data_{args.data}",
    f"dataset_{args.dataset}",
]

# Add transformer details if enabled
if args.do_transformer:
    out_dir_components.extend(
        [
            f"transformer_{args.transformer_version}",
            f"depth_{args.transformer_depth}",
        ]
    )

# Add encoding details if enabled
if args.add_encodings:
    out_dir_components.append(f"encoding_{args.encodings}")
    if args.encodings == "RW":
        out_dir_components.append(f"rwtype_{args.random_walk_type}")
    elif args.encodings == "LCP":
        out_dir_components.append(f"curvature_{args.curvature_type}")
    elif args.encodings == "Laplacian":
        out_dir_components.append(f"laptype_{args.laplacian_type}")

# Add timestamp for uniqueness
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir_components.append(f"time_{timestamp}")

# Construct the path
out_dir = path.Path("_".join(out_dir_components))

# Create directory
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.makedirs_p()
baselogger = get_logger("base logger", f"{out_dir}/logging.log", not args.nostdout)
resultlogger = get_logger("result logger", f"{out_dir}/result.log", not args.nostdout)
baselogger.info(args)

resultlogger.info(args)


# Keep track of the test accuracy for the best val accuracy
overall_test_accuracies_best_val: list[float] = []

seed: int
for seed in range(1, 9):
    seed += 1
    print(f"The seed is {seed}")
    # gpu, seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    #### configure output directory

    val_idx: list[int]
    test_idx: list[int]
    train_idx: torch.Tensor

    for run in range(1, args.n_runs + 1):
        run_dir = out_dir / f"{seed}_{run}"
        run_dir.makedirs_p()

        # load data
        args.split = run
        _, train_idx, test_idx = load(args)
        val_idx, test_idx = get_split(Y[test_idx], 0.2)
        # I believe train is fixed across all runs.
        train_idx = torch.LongTensor(train_idx).to(device)
        val_idx = torch.LongTensor(val_idx).to(device)
        test_idx = torch.LongTensor(test_idx).to(device)

        # model
        model, optimizer = initialise(X, Y, G, args)
        model = model.to(device)
        X = X.to(device)
        V = V.to(device)
        E = E.to(device)
        Y = Y.to(device)

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
        test_accs_for_best_val: list[float] = (
            []
        )  # List to store test accuracy for the best validation accuracy
        train_accs: list[float] = []
        val_accs: list[float] = []
        test_accs: list[float] = []
        for epoch in range(args.epochs):
            # train
            tic_epoch = time.time()
            model.train()

            optimizer.zero_grad()
            Z = model(X, V, E)  # this call forward.
            loss = F.nll_loss(Z[train_idx], Y[train_idx])

            loss.backward()
            optimizer.step()

            train_time = time.time() - tic_epoch

            # eval
            model.eval()
            Z: torch.Tensor = model(X, V, E)  # this calls forward

            # gets the trains, test and val accuracy
            train_acc: float = accuracy(Z[train_idx], Y[train_idx])
            test_acc: float = accuracy(Z[test_idx], Y[test_idx])
            val_acc: float = accuracy(Z[val_idx], Y[val_idx])

            # Store accuracies
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            test_accs.append(test_acc)

            # log acc
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                bad_counter = 0
                test_accs_for_best_val.append(
                    test_acc
                )  # Save the test accuracy when validation accuracy improves
            else:
                bad_counter += 1
                if bad_counter >= args.patience:
                    break
            if epoch % 20 == 0:
                baselogger.info(
                    f"epoch:{epoch} | loss:{loss:.4f} | train acc:{train_acc:.2f} | val acc:{val_acc:.2f} | test_acc_best_val: {test_accs_for_best_val[-1]:.2f}  | best_test_acc: {best_test_acc:.2f} | test acc:{test_acc:.2f} | time:{train_time*1000:.1f}ms"
                )

        # Store accuracies with unique key
        run_key = f"seed_{seed}_run_{run}"
        all_results["train_accs"][run_key] = train_accs
        all_results["val_accs"][run_key] = val_accs
        all_results["test_accs"][run_key] = test_accs
        all_results["params"][run_key] = {
            "model": model_name,
            "nlayer": nlayer,
            "dataset": dataname,
            "encodings": args.encodings if args.add_encodings else "none",
            "transformer": args.do_transformer,
            "final_train_acc": train_accs[-1],
            "final_val_acc": val_accs[-1],
            "final_test_acc": test_accs[-1],
            "best_val_acc": best_val_acc,
            "best_test_acc": best_test_acc,
        }

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


# def create_summary_plots(all_results, out_dir):
#     """Create summary plots from all runs."""
#     plt.figure(figsize=(15, 10))

#     # Plot 1: Box plot of final accuracies
#     plt.subplot(2, 2, 1)
#     final_trains = [
#         params["final_train_acc"] for params in all_results["params"].values()
#     ]
#     final_vals = [params["final_val_acc"] for params in all_results["params"].values()]
#     final_tests = [
#         params["final_test_acc"] for params in all_results["params"].values()
#     ]

#     plt.boxplot(
#         [final_trains, final_vals, final_tests], labels=["Train", "Validation", "Test"]
#     )
#     plt.title("Distribution of Final Accuracies")
#     plt.ylabel("Accuracy (%)")

#     # Plot 2: Learning curves with confidence intervals
#     plt.subplot(2, 2, 2)

#     # Calculate max epochs correctly
#     max_epochs = max(len(accs) for accs in all_results["train_accs"].values())
#     # Create epochs array starting from 0 to max_epochs-1
#     epochs = range(max_epochs)  # Changed from range(1, max_epochs + 1)

#     # Helper function to pad and calculate statistics
#     def get_stats(accs_dict):
#         # Pad sequences to the same length
#         padded_accs = np.array(
#             [
#                 accs + [accs[-1]] * (max_epochs - len(accs))
#                 for accs in accs_dict.values()
#             ]
#         )
#         return np.mean(padded_accs, axis=0), np.std(padded_accs, axis=0)

#     # Calculate statistics
#     train_mean, train_std = get_stats(all_results["train_accs"])
#     val_mean, val_std = get_stats(all_results["val_accs"])
#     test_mean, test_std = get_stats(all_results["test_accs"])

#     # Verify shapes before plotting
#     assert len(epochs) == len(
#         train_mean
#     ), f"Epochs length: {len(epochs)}, Train mean length: {len(train_mean)}"
#     assert len(epochs) == len(
#         val_mean
#     ), f"Epochs length: {len(epochs)}, Val mean length: {len(val_mean)}"
#     assert len(epochs) == len(
#         test_mean
#     ), f"Epochs length: {len(epochs)}, Test mean length: {len(test_mean)}"

#     # Plot with confidence intervals
#     plt.plot(epochs, train_mean, "b-", label="Train")
#     plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2)
#     plt.plot(epochs, val_mean, "g-", label="Validation")
#     plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2)
#     plt.plot(epochs, test_mean, "r-", label="Test")
#     plt.fill_between(epochs, test_mean - test_std, test_mean + test_std, alpha=0.2)

#     plt.title("Average Learning Curves (with std)")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy (%)")
#     plt.legend()

#     # Save results and plot
#     results_dir = out_dir / "results"
#     results_dir.makedirs_p()

#     # Save numerical results
#     with open(results_dir / "all_results.pkl", "wb") as f:
#         pickle.dump(all_results, f)

#     # Save plot
#     plt.tight_layout()
#     plt.savefig(results_dir / "summary_plots.png", dpi=300, bbox_inches="tight")
#     plt.close()

#     print("\n=== Results Saved ===")
#     print(f"All results saved to: {results_dir / 'all_results.pkl'}")
#     print(f"Summary plots saved to: {results_dir / 'summary_plots.png'}")


# Call the plotting function at the end
# create_summary_plots(all_results, out_dir)
