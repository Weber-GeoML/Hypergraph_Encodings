from random import sample

import numpy as np
import torch


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
