"""Split data into train, val, test sets."""

from random import sample

import torch

# load data

### configure logger


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
