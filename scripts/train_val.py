""" File taken from UniGCN"""

import datetime
import os
import shutil
import time
from random import sample

import config
import numpy as np
import path
import torch
import torch.nn.functional as F
from torch.optim import optimizer

# load data
from encodings_hnns.data_handling import load

### configure logger
from uniGCN.logger import get_logger
from uniGCN.prepare import fetch_data, initialise, accuracy

# File originally taken from UniGCN repo

# Check if CUDA is available and move tensors to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_accs: list = []
best_val_accs: list = []
best_test_accs: list = []


args = config.parse()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

use_norm: str = "use-norm" if args.use_norm else "no-norm"
add_self_loop: str = "add-self-loop" if args.add_self_loop else "no-self-loop"

dataname: str = f"{args.data}_{args.dataset}"
model_name: str = args.model_name
nlayer: int = args.nlayer
dirname = f"{datetime.datetime.now()}".replace(" ", "_").replace(":", ".")


# load data
X: torch.Tensor
Y: torch.Tensor
print(f"We are adding the {args.encodings} encodings")
X, Y, G = fetch_data(
    args,
    add_encodings=args.add_encodings,
    encodings=args.encodings,
    laplacian_type=args.laplacian_type,
    random_walk_type=args.random_walk_type,
    k_rw=args.k_rw,
    curvature_type=args.curvature_type,
)
print(f"X are the features \n {X} \n with shape {X.shape}")
features_shape = X.shape[0]
print(f"Y are the labels \n {Y}")
print(f"G is the hg")


def get_split(Y, p: float = 0.2) -> tuple[list[int], list[int]]:
    """TODO

    Args:
        Y:
        p:

    """
    Y: list = Y.tolist()
    N: int = len(Y)
    nclass: int = len(set(Y))
    D: list = [[] for _ in range(nclass)]
    for i, y in enumerate(Y):
        D[y].append(i)
    k: int = int(N * p / nclass)
    val_idx: list[int] = torch.cat(
        [torch.LongTensor(sample(idxs, k)) for idxs in D]
    ).tolist()
    test_idx: list[int] = list(set(range(N)) - set(val_idx))

    return val_idx, test_idx


out_dir: str = path.Path(f"./{args.out_dir}/{model_name}_{nlayer}_{dataname}/")
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.makedirs_p()
baselogger = get_logger("base logger", f"{out_dir}/logging.log", not args.nostdout)
resultlogger = get_logger("result logger", f"{out_dir}/result.log", not args.nostdout)
baselogger.info(args)

resultlogger.info(args)


# Keep track of the test accuracy for the best val accuracy
overall_test_accuracies_best_val: list = []

for seed in range(1, 9):
    seed += 1
    print(f"The seed is {seed}")
    # gpu, seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    #### configure output directory

    out_dir: str = path.Path(
        f"./{args.out_dir}/{model_name}_{nlayer}_{dataname}/seed_{seed}"
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
        val_idx, test_idx = get_split(Y[test_idx], 0.2)
        train_idx: torch.Tensor = torch.LongTensor(train_idx).to(device)
        val_idx: torch.Tensor = torch.LongTensor(val_idx).to(device)
        test_idx: torch.Tensor = torch.LongTensor(test_idx).to(device)

        # model
        model, optimizer = initialise(X, Y, G, args)

        baselogger.info(f"Run {run}/{args.n_runs}, Total Epochs: {args.epochs}")
        baselogger.info(model)
        baselogger.info(
            f"total_params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        tic_run = time.time()

        best_val_acc, best_test_acc, test_acc, Z, bad_counter = 0, 0, 0, None, 0
        test_accs_for_best_val = (
            []
        )  # List to store test accuracy for the best validation accuracy
        for epoch in range(args.epochs):
            # train
            tic_epoch = time.time()
            model.train()

            optimizer.zero_grad()
            Z = model(X)
            loss = F.nll_loss(Z[train_idx], Y[train_idx])

            loss.backward()
            optimizer.step()

            train_time = time.time() - tic_epoch

            # eval
            model.eval()
            Z = model(X)
            train_acc = accuracy(Z[train_idx], Y[train_idx])
            test_acc = accuracy(Z[test_idx], Y[test_idx])
            val_acc = accuracy(Z[val_idx], Y[val_idx])

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
