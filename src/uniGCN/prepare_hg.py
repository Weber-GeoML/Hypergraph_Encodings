import scipy.sparse as sp
import torch
import torch.optim as optim
import torch_sparse
from torch_scatter import scatter
import numpy as np
import argparse
from typing import Union

from uniGCN.UniGCN import UniGNN

# Check if CUDA is available and move tensors to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialise_for_hypergraph_classification(
    list_hg: list[dict[str, Union[dict, np.ndarray, int]]],
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict, dict, dict]:
    """Initializes model and optimizer for hypergraph classification.

    Unlike node classification where we have one large hypergraph,
    here we have multiple smaller hypergraphs, each with its own label.

    Args:
        list_hg: List of hypergraph dictionaries, each containing:
            - hypergraph: dict[str, list[int]] mapping edge names to node lists
            - features: np.ndarray of node features
            - labels: int representing hypergraph class
            - n: int number of nodes
        args: Configuration arguments

    Returns:
        tuple containing:
        - model: The neural network model
        - optimizer: The optimizer
        - degVs:
        - degEs:
        - degE2s:
    """

    Y: list[int] = []
    degVs: dict = {}
    degEs: dict = {}
    degE2s: dict = {}

    # Validate input hypergraphs
    required_keys = {"hypergraph", "features", "labels", "n"}
    for idx, hg in enumerate(list_hg):
        if not all(key in hg for key in required_keys):
            missing = required_keys - set(hg.keys())
            raise ValueError(f"Hypergraph {idx} missing required keys: {missing}")

        if not isinstance(hg["labels"], (int, np.integer)):
            raise ValueError(
                f"Hypergraph {idx} label must be integer, got {type(hg['labels'])}"
            )

    # Verify all hypergraphs have same feature dimension
    feature_dims = [hg["features"].shape[1] for hg in list_hg]
    if len(set(feature_dims)) > 1:
        raise ValueError(
            f"Inconsistent feature dimensions across hypergraphs: {feature_dims}"
        )

    for idx, hg in enumerate(list_hg):
        X: torch.Tensor
        Y: torch.Tensor
        G: dict
        X = hg["features"]
        G = hg["hypergraph"]
        Y.append(hg["labels"])
        # dis
        N, M = X.shape[0], len(G)
        indptr: list[int]
        indices: list[int]
        data: list[int]
        indptr, indices, data = [0], [], []
        # loop through G items, which are keys: edges, values edge as a list that contains the vertices
        for e, vs in G.items():
            indices += vs
            data += [1] * len(vs)  # extend data by adding as many '1's as there are vs
            indptr.append(
                len(indices)
            )  # keep track of the number of vertices in each edge

        # csc_matrix((data, indices, indptr), [shape=(M, N)])
        # is the standard CSC representation where the row indices for
        # column i are stored in ``indices[indptr[i]:indptr[i+1]]``
        # and their corresponding values are stored in
        # ``data[indptr[i]:indptr[i+1]]``.  If the shape parameter is
        # not supplied, the matrix dimensions are inferred from
        # the index arrays.
        # More explanations:
        # csc_matrix((data, indices, indptr), [shape=(M, N)])
        # This represents a sparse matrix in Compressed Sparse Column (CSC) format.
        # 'data' contains the non-zero values of the matrix.
        # 'indices' contains the row indices corresponding to each non-zero value in 'data'.
        # 'indptr' is an array that specifies where each column starts in 'data' and 'indices'.
        # The entries for column i are found between indptr[i] and indptr[i+1] in 'data' and 'indices'.
        # eg
        # Construct the spare matrix:
        # 0  10  0
        # 20 0   30
        # 0  40  50
        # 60 0   0
        # data = [20, 60, 10, 40, 30, 50]  # (column 1: 20, 60), (column 2: 10, 40), (column 3: 30, 50)
        # indices: The corresponding row indices for each non-zero element:
        # indices = [1, 3, 0, 2, 1, 2]
        # indptr: Array indicating the start of each column in the data array:
        # Column 0 starts at index 0 in data.
        # Column 1 starts at index 2 in data.
        # Column 2 starts at index 4 in data
        # indptr = [0, 2, 4, 6]
        # N, M = 4, 3  # Shape of the matrix (4 rows, 3 columns)
        # H = sp.csc_matrix((data, indices, indptr), shape=(N, M), dtype=int).tocsr()
        H = sp.csc_matrix(
            (data, indices, indptr), shape=(N, M), dtype=int
        ).tocsr()  # V x E
        print(f"H is \n {H}")
        degV: torch.Tensor = (
            torch.from_numpy(H.sum(1)).view(-1, 1).float()
        )  # the degree of each vertices
        degE2: torch.Tensor = (
            torch.from_numpy(H.sum(0)).view(-1, 1).float()
        )  # the degree of each edge
        # Say:
        # [[0 1 0]
        #  [2 0 3]
        #  [0 4 0]]
        # then:
        # (row, col), value = torch_sparse.from_scipy(H)
        # are as follow
        # Row indices: tensor([0, 1, 1, 2])  # Rows of non-zero elements
        # Column indices: tensor([1, 0, 2, 1])  # Columns of non-zero elements
        # Values: tensor([1, 2, 3, 4])  # Non-zero values at corresponding (row, col)
        (row, col), value = torch_sparse.from_scipy(H)
        V, E = row, col
        # eg: vertex 0 is in edge 1,
        # vertex 1 is in edge 0
        # vertex 1 is in edge 2
        # vertex 2 is in edge 1
        # taking the example above
        assert args.first_aggregate in (
            "mean",
            "sum",
        ), "use `mean` or `sum` for first-stage aggregation"
        degE = scatter(degV[V], E, dim=0, reduce=args.first_aggregate)
        # this is what goes into the UniGCN/UniGCNII formula
        # for d_i and d_e
        # x_i= 1/√d_i sum 1/√d_e Wh_e,
        degE: torch.Tensor = degE.pow(-0.5)
        degV: torch.Tensor = degV.pow(-0.5)
        degV[degV.isinf()] = (
            1  # when not added self-loop, some nodes might not be connected with any edge
        )

        degVs[idx] = degV.to(device)
        degEs[idx] = degE.to(device)
        degE2s[idx] = degE2.pow(-1.0).to(device)

    # Move all tensors to same device
    Y_tensor = torch.tensor(Y, device=device)
    for idx in degVs:
        degVs[idx] = degVs[idx].to(device)
        degEs[idx] = degEs[idx].to(device)
        degE2s[idx] = degE2s[idx].to(device)

    # nfeat: the dimension of the features
    # nclass: the number of classes in the labels
    nfeat: int
    nclass: int
    nfeat: int = X.shape[1]
    Y_tensor = torch.tensor(Y)
    nclass = len(Y_tensor.unique())
    nlayer: int = args.nlayer
    nhid: int = args.nhid
    nhead: int = args.nhead

    # Model initialization
    supported_models = {"UniGCN", "UniGAT", "UniGIN", "UniSAGE"}
    if args.model_name not in supported_models:
        raise ValueError(
            f"Model {args.model_name} not supported. Must be one of: {supported_models}"
        )

    # UniGNN and optimiser
    if args.model_name == "UniGCNII":
        raise NotImplementedError
    if args.model_name in ["UniGCN", "UniGAT", "UniGIN", "UniSAGE"]:
        # include UniGCN, UniGAT, UniGIN, UniSAGE
        model = UniGNN(args, nfeat, nhid, nclass, nlayer, nhead)
        optimiser = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    else:
        raise NotImplementedError

    model.to(device)
    print("Managed to get the model to the device")
    print(f"Initializing {args.model_name} for hypergraph classification")
    print(f"Number of hypergraphs: {len(list_hg)}")
    print(f"Feature dimension: {nfeat}")
    print(f"Number of classes: {nclass}")
    return model, optimiser, degVs, degEs, degE2s
