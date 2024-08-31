""" File taken from https://github.com/RaphaelPellegrin/UniGNN/tree/master"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch_sparse
from torch_scatter import scatter

from encodings_hnns.data_handling import load
from encodings_hnns.encodings import HypergraphEncodings
from uniGCN.HyperGCN import HyperGCN
from uniGCN.UniGCN import UniGCNII, UniGNN

# Check if CUDA is available and move tensors to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(Z, Y):
    """TODO

    Args:
        Z:
            TODO
        Y:
            TODO
    """
    return 100 * Z.argmax(1).eq(Y).float().mean().item()


def fetch_data(
    args,
    add_encodings: bool = False,
    encodings: str = "RW",
    laplacian_type: str = "Hodge",
    random_walk_type: str = "WE",
    k_rw: int = 20,
    curvature_type: str = "ORC",
    normalize_features: bool = False,
    normalize_encodings: bool = False,
):
    """Gets the data

    Args:
        args:
            TODO
        add_encodings: added by RP!
            wehther to add an encoding or not
        encodings: added by RP!
            which encoding to add
            Can be RW, Laplacian, LCP, LDP
        laplacian_type:
            Which laplacian to use
            Hodge or Normalized
        random_walk_type:
            which rw to use
            WE or EN or EE
        k_rw:
            number of hops for RW
            default is 20
        curvature_type:
            ORC or FRC
        normalize_features:
            whether to normalize the features
        normalize_encodings:
            whether to normalize the encodings
    """
    dataset, _, _ = load(args)
    args.dataset_dict = dataset
    dataset_name: str = f"{args.data}_{args.dataset}"

    shape_before = dataset["features"].shape
    print(f"The features shape are {shape_before}")

    # No normalization
    if not normalize_features:
        normalize_encodings = False
        # added by RP!
        if add_encodings:
            print("We are adding encodings!")
            hgencodings = HypergraphEncodings()
            if encodings == "RW":
                print("Adding the RW encodings")
                dataset = hgencodings.add_randowm_walks_encodings(
                    dataset,
                    rw_type=random_walk_type,
                    k=k_rw,
                    normalized=True,
                    dataset_name=dataset_name,
                )  # normalized is True because of some formatting. normalized is just whether to add [] around festures/encodings...
            elif encodings == "Laplacian":
                print("Adding the Laplacian encodings")
                dataset = hgencodings.add_laplacian_encodings(
                    dataset,
                    type=laplacian_type,
                    normalized=True,
                    dataset_name=dataset_name,
                )
            elif encodings == "LCP":
                print("Adding the LCP encodings")
                dataset = hgencodings.add_curvature_encodings(
                    dataset,
                    type=curvature_type,
                    normalized=True,
                    dataset_name=dataset_name,
                )
            elif encodings == "LDP":
                print("Adding the LDP encodings")
                dataset = hgencodings.add_degree_encodings(
                    dataset, normalized=True, dataset_name=dataset_name
                )

            print(f"The features are {dataset['features']}")
            shape_after = dataset["features"].shape
            print(f"The features shape are {dataset['features'].shape}")
            # use the toy hypergraph
            # check that the features are added there
            assert (
                shape_before[0] == shape_after[0]
            ), f"The shape are {shape_before} and {shape_after}"
            assert (
                shape_before[1] != shape_after[1]
            ), f"The shape are {shape_before} and {shape_after}"
        # nothing has been normalized
        X, Y, G = dataset["features"], dataset["labels"], dataset["hypergraph"]
        # node features in sparse representation
        X = sp.csr_matrix(X, dtype=np.float32)
        X = torch.FloatTensor(np.array(X.todense()))

    # normalize the features
    elif normalize_features == True:
        if normalize_encodings or not add_encodings:
            # added by RP!
            if add_encodings:
                print("We are adding encodings!")
                hgencodings = HypergraphEncodings()
                if encodings == "RW":
                    print("Adding the RW encodings")
                    dataset = hgencodings.add_randowm_walks_encodings(
                        dataset,
                        rw_type=random_walk_type,
                        k=k_rw,
                        normalized=True,
                        dataset_name=dataset_name,
                    )
                elif encodings == "Laplacian":
                    print("Adding the Laplacian encodings")
                    dataset = hgencodings.add_laplacian_encodings(
                        dataset,
                        type=laplacian_type,
                        normalized=True,
                        dataset_name=dataset_name,
                    )
                elif encodings == "LCP":
                    print("Adding the LCP encodings")
                    dataset = hgencodings.add_curvature_encodings(
                        dataset,
                        type=curvature_type,
                        normalized=True,
                        dataset_name=dataset_name,
                    )
                elif encodings == "LDP":
                    print("Adding the LDP encodings")
                    dataset = hgencodings.add_degree_encodings(
                        dataset, normalized=True, dataset_name=dataset_name
                    )

                print(f"The features are {dataset['features']}")
                shape_after = dataset["features"].shape
                print(f"The features shape are {dataset['features'].shape}")
                # use the toy hypergraph
                # check that the features are added there
                assert (
                    shape_before[0] == shape_after[0]
                ), f"The shape are {shape_before} and {shape_after}"
                assert (
                    shape_before[1] != shape_after[1]
                ), f"The shape are {shape_before} and {shape_after}"
            X, Y, G = dataset["features"], dataset["labels"], dataset["hypergraph"]

            # normalize everything here (features and encodings, if added)
            # node features in sparse representation
            X = sp.csr_matrix(normalise(np.array(X)), dtype=np.float32)
            X = torch.FloatTensor(np.array(X.todense()))

        # normalize the features, but not the encodings
        elif not normalize_encodings:
            # normalize here
            dataset["features"] = normalise(np.array(dataset["features"]))
            shape_after = dataset["features"].shape
            print(f"The features shape are {shape_after}")

            # added by RP!
            if add_encodings:
                print("We are adding encodings!")
                hgencodings = HypergraphEncodings()
                if encodings == "RW":
                    print("Adding the RW encodings")
                    dataset = hgencodings.add_randowm_walks_encodings(
                        dataset,
                        rw_type=random_walk_type,
                        k=k_rw,
                        normalized=normalize_encodings,
                        dataset_name=dataset_name,
                    )
                elif encodings == "Laplacian":
                    print("Adding the Laplacian encodings")
                    dataset = hgencodings.add_laplacian_encodings(
                        dataset,
                        type=laplacian_type,
                        normalized=normalize_encodings,
                        dataset_name=dataset_name,
                    )
                elif encodings == "LCP":
                    print("Adding the LCP encodings")
                    dataset = hgencodings.add_curvature_encodings(
                        dataset,
                        type=curvature_type,
                        normalized=normalize_encodings,
                        dataset_name=dataset_name,
                    )
                elif encodings == "LDP":
                    print("Adding the LDP encodings")
                    dataset = hgencodings.add_degree_encodings(
                        dataset,
                        normalized=normalize_encodings,
                        dataset_name=dataset_name,
                    )

                print(f"The features are {dataset['features']}")
                shape_after = dataset["features"].shape
                print(f"The features shape are {dataset['features'].shape}")
                # use the toy hypergraph
                # check that the features are added there
                assert (
                    shape_before[0] == shape_after[0]
                ), f"The shape are {shape_before} and {shape_after}"
                assert (
                    shape_before[1] != shape_after[1]
                ), f"The shape are {shape_before} and {shape_after}"
            # encodings are unormalized
            X, Y, G = dataset["features"], dataset["labels"], dataset["hypergraph"]
            # node features in sparse representation
            X = sp.csr_matrix(X, dtype=np.float32)
            X = torch.FloatTensor(np.array(X.todense()))

    # labels
    Y = np.array(Y)
    Y = torch.LongTensor(np.where(Y)[1])

    X, Y = X.to(device), Y.to(device)
    return X, Y, G


def initialise(X, Y, G, args, unseen=None) -> tuple:
    """Initialises model, optimiser, normalises graph, and features

    Arguments:
        X, Y, G: the entire dataset (with graph, features, labels)
    args: arguments
    unseen: if not None, remove these nodes from hypergraphs

    Returns:
        a tuple with model details (UniGNN, optimiser)
    """

    G = G.copy()

    if unseen is not None:
        unseen = set(unseen)
        # remove unseen nodes
        for e, vs in G.items():
            G[e] = list(set(vs) - unseen)

    if args.add_self_loop:
        Vs = set(range(X.shape[0]))

        # only add self-loop to those are orginally un-self-looped
        # TODO:maybe we should remove some repeated self-loops?
        for edge, nodes in G.items():
            if len(nodes) == 1 and nodes[0] in Vs:
                Vs.remove(nodes[0])

        for v in Vs:
            G[f"self-loop-{v}"] = [v]

    N, M = X.shape[0], len(G)
    indptr, indices, data = [0], [], []
    for e, vs in G.items():
        indices += vs
        data += [1] * len(vs)
        indptr.append(len(indices))
    H = sp.csc_matrix((data, indices, indptr), shape=(N, M), dtype=int).tocsr()  # V x E
    degV = torch.from_numpy(H.sum(1)).view(-1, 1).float()
    degE2 = torch.from_numpy(H.sum(0)).view(-1, 1).float()

    (row, col), value = torch_sparse.from_scipy(H)
    V, E = row, col

    assert args.first_aggregate in (
        "mean",
        "sum",
    ), "use `mean` or `sum` for first-stage aggregation"
    degE = scatter(degV[V], E, dim=0, reduce=args.first_aggregate)
    degE = degE.pow(-0.5)
    degV = degV.pow(-0.5)
    degV[degV.isinf()] = (
        1  # when not added self-loop, some nodes might not be connected with any edge
    )

    V, E = V.to(device), E.to(device)
    args.degV = degV.to(device)
    args.degE = degE.to(device)
    args.degE2 = degE2.pow(-1.0).to(device)

    nfeat, nclass = X.shape[1], len(Y.unique())
    nlayer = args.nlayer
    nhid = args.nhid
    nhead = args.nhead

    # UniGNN and optimiser
    if args.model_name == "UniGCNII":
        model = UniGCNII(args, nfeat, nhid, nclass, nlayer, nhead, V, E)
        optimiser = torch.optim.Adam(
            [
                dict(params=model.reg_params, weight_decay=0.01),
                dict(params=model.non_reg_params, weight_decay=5e-4),
            ],
            lr=0.01,
        )
    elif args.model_name == "HyperGCN":
        args.fast = True
        dataset = args.dataset_dict
        model = HyperGCN(
            args,
            nfeat,
            nhid,
            nclass,
            nlayer,
            dataset["n"],
            dataset["hypergraph"],
            dataset["features"],
        )
        optimiser = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    else:
        model = UniGNN(args, nfeat, nhid, nclass, nlayer, nhead, V, E)
        optimiser = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.to(device)
    print("Managed to get the model to the device")

    return model, optimiser


def normalise(M):
    """row-normalise sparse matrix

    Arguments:
        M:
            scipy sparse matrix

    Returns:
        D^{-1} M
            where D is the diagonal node-degree matrix
    """

    d = np.array(M.sum(1))

    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.0
    DI = sp.diags(di)  # D inverse i.e. D^{-1}

    return DI.dot(M)
