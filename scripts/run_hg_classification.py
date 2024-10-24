""" Equivalent of train_val.py, but for hypergraph classification"""

import datetime
import os
import pickle
import shutil
import time
from random import sample

import numpy as np
import path
import torch
import torch.nn.functional as F

# load data
# load data
from encodings_hnns.data_handling import load
from torch.optim import optimizer

### configure logger
### configure logger
from uniGCN.logger import get_logger
from uniGCN.prepare import accuracy, calculate_V_E
from uniGCN.prepare_hg import initialise_for_hypergraph_classification

import config

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


# load the data from data\hypergraph_classification_datasets/imdb_hypergraphs.pickle
with open("data/hypergraph_classification_datasets/imdb_hypergraphs.pickle", "rb") as f:
    imdb = pickle.load(f)

with open(
    "data/hypergraph_classification_datasets/collab_hypergraphs.pickle", "rb"
) as f:
    collab = pickle.load(f)

with open(
    "data/hypergraph_classification_datasets/reddit_hypergraphs.pickle", "rb"
) as f:
    reddit = pickle.load(f)

# each dataset is a list of dicts that contrains the hypergraph, the features, the labels and the number of nodes
datasets: dict[list[dict[dict, np.matrix, np.ndarray, int]]] = {
    "imdb": imdb,
    "collab": collab,
    "reddit": reddit,
}


# split the data
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


# Might have to do something more along these lines:
# def get_split(Y, p: float = 0.2) -> tuple[list[int], list[int]]:
#     """Splits Y into a test and val set.

#     Args:
#         Y:
#             the labels of nodes.
#         p:
#             the proportion of nodes in the val set

#     Returns:
#         val_idx:
#             the indices of nodes in the val set
#         test_idx:
#             the indices of nodes in the test set

#     """
#     Y: list = Y.tolist()
#     N: int = len(Y)  # number of nodes
#     nclass: int = len(set(Y))  # number of different labels
#     D: list = [[] for _ in range(nclass)]
#     for i, y in enumerate(Y):
#         D[y].append(i)
#     k: int = int(N * p / nclass)
#     val_idx: list[int] = torch.cat(
#         [torch.LongTensor(sample(idxs, k)) for idxs in D]
#     ).tolist()
#     test_idx: list[int] = list(set(range(N)) - set(val_idx))

#     return val_idx, test_idx


# Now same as previous file:
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
features_shape = imdb[0]["features"].shape


out_dir: str = path.Path(f"./{args.out_dir}/{model_name}_{nlayer}_hg_classification/")
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

    out_dir: str = path.Path(
        f"./{args.out_dir}/{model_name}_{nlayer}_hg_classification/seed_{seed}"
    )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.makedirs_p()

    for run in range(1, args.n_runs + 1):
        run_dir = out_dir / f"{run}"
        run_dir.makedirs_p()

        # load data
        args.split = run
        _, train_idx, test_idx = load(args)
        val_idx: list[int]
        test_idx: list[int]
        val_idx, test_idx = get_split(Y[test_idx], 0.2)
        train_idx: torch.Tensor = torch.LongTensor(train_idx).to(device)
        val_idx: torch.Tensor = torch.LongTensor(val_idx).to(device)
        test_idx: torch.Tensor = torch.LongTensor(test_idx).to(device)

        # model
        model, optimizer = initialise_for_hypergraph_classification(
            imdb, args
        )  # using imdb as an example for now

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
            if epoch % 20 == 0:
                baselogger.info(
                    f"epoch:{epoch} | loss:{loss:.4f} | train acc:{train_acc:.2f} | val acc:{val_acc:.2f} | test_acc_best_val: {test_accs_for_best_val[-1]:.2f}  | best_test_acc: {best_test_acc:.2f} | test acc:{test_acc:.2f} | time:{train_time*1000:.1f}ms"
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
