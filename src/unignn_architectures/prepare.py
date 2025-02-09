"""File taken from https://github.com/RaphaelPellegrin/UniGNN/tree/master"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim
import torch_sparse
from torch_scatter import scatter

from encodings_hnns.data_handling import load
from encodings_hnns.encodings import HypergraphEncodings
from unignn_architectures.HyperGCN import HyperGCN
from unignn_architectures.UniGCN import UniGCNII, UniGNN

# Check if CUDA is available and move tensors to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(Z: torch.Tensor, Y: torch.Tensor) -> float:
    """Computes the accuracy between prediction Z and true labels
    Y.

    Args:
        Z:
            predictions Z. As a vector or probabilities
        Y:
            the labels

    Returns:
        the accuracy between the prediction Z and the labels
        Y.
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
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Gets the data.

    Also adds the encodings, if specified
    by the args.

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

    Returns:
        X:
            the features
        Y:
            the labels
        G:
            the whole hypergraph
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
        if not add_encodings:
            print("We are not adding any encodings")
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
                    laplacian_type=laplacian_type,
                    normalized=True,
                    dataset_name=dataset_name,
                )
            elif encodings == "LCP":
                print("Adding the LCP encodings")
                dataset = hgencodings.add_curvature_encodings(
                    dataset,
                    curvature_type=curvature_type,
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
    elif normalize_features:
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
                        laplacian_type=laplacian_type,
                        normalized=True,
                        dataset_name=dataset_name,
                    )
                elif encodings == "LCP":
                    print("Adding the LCP encodings")
                    dataset = hgencodings.add_curvature_encodings(
                        dataset,
                        curvature_type=curvature_type,
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
            X, Y, G = (
                dataset["features"],
                dataset["labels"],
                dataset["hypergraph"],
            )

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
                        laplacian_type=laplacian_type,
                        normalized=normalize_encodings,
                        dataset_name=dataset_name,
                    )
                elif encodings == "LCP":
                    print("Adding the LCP encodings")
                    dataset = hgencodings.add_curvature_encodings(
                        dataset,
                        curvature_type=curvature_type,
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
            X, Y, G = (
                dataset["features"],
                dataset["labels"],
                dataset["hypergraph"],
            )
            # node features in sparse representation
            X = sp.csr_matrix(X, dtype=np.float32)
            X = torch.FloatTensor(np.array(X.todense()))

    # labels
    Y = np.array(Y)
    Y = torch.LongTensor(np.where(Y)[1])

    X, Y = X.to(device), Y.to(device)
    return X, Y, G


def initialise(
    X: torch.Tensor, Y: torch.Tensor, G: dict, args, unseen: None = None
) -> tuple:
    """Initialises model, optimiser, normalises graph, and features

    Arguments:
        X:
            the features
        Y:
            the labels
        G:
            the graph
        args:
            arguments
        unseen:
            if not None, remove these nodes from hypergraphs
            This is typically used for:
            Train/Test Split: When you want to evaluate how well your model generalizes to unseen nodes
            Training: Use a subset of nodes
            Testing: Use the remaining "unseen" nodes
            Cross-Validation: When you want to validate your model on different subsets of nodes
            Training: Use most nodes
            Validation: Use some "unseen" nodes to tune hyperparameters
            Inductive Learning: Testing the model's ability to handle new nodes that weren't present during training
            Training: Learn patterns from known nodes
            Testing: Apply learned patterns to completely new "unseen" nodes

    Returns:
        a tuple with model details (UniGNN, optimiser)
    """

    G = G.copy()

    if unseen is not None:
        unseen = set(unseen)
        # remove unseen nodes
        for (
            e,
            vs,
        ) in G.items():  # loops through edges and the vertices they contain
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

    N: int  # number of nodes, number of rows
    M: int  # number of edges, number of columns
    N, M = X.shape[0], len(G)
    indptr: list[int]
    indices: list[int]  # the row indices for each element of data.
    data: list[int]  # keeps track of the values of non-zero elements of H
    # they are all 1.
    indptr, indices, data = [0], [], []
    # loop through G items, which are keys: edges, values edge as a list that contains the vertices
    for e, vs in G.items():
        indices += vs  # the corresponding row indice, as nodes are rows
        # this keep tracks of the non-zero elements in the final matrix H
        data += [1] * len(vs)  # extend data by adding as many '1's as there are vs
        # the matrix H only contain 1s!
        indptr.append(len(indices))  # keep track of the number of vertices in each edge
        # this is just used to tell use where column begin and start in data

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
    H = sp.csc_matrix((data, indices, indptr), shape=(N, M), dtype=int).tocsr()  # V x E
    print(f"H is \n {H}")  # this is just the incidence matrix

    # Calculate the degree of each vertex (degree_vertices) by summing the values
    # in each row of the matrix H.
    # This gives the number of edges connected to each vertex.
    # The result is converted from NumPy to a PyTorch tensor, reshaped into a
    # column vector (N x 1), and cast to float.
    degree_vertices: torch.Tensor = (
        torch.from_numpy(H.sum(1)).view(-1, 1).float()
    )  # the degree of each vertices

    # Similarly, calculate the degree of each edge (dege2) by summing
    # the values in each column of the matrix H.
    # This gives the number of vertices connected to each edge.
    # Again, the result is converted from NumPy to a PyTorch tensor,
    # reshaped, and cast to float.
    dege2: torch.Tensor = (
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
    # V represents the row indices (i.e., vertices), and E represents
    # the column indices (i.e., edges) of the non-zero elements in H.
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
    # So, degree_vertices[V] selects values from degree_vertices at positions indicated by V.
    # This operation retrieves the degree values corresponding to specific vertices.
    # degree_edges represents aggregated degree values of edges after combining the corresponding vertices'
    # degree values.
    # this matches this from the paper:
    #
    # where we define de = (1/|e|) sum di as the average degree
    # of a hyperedge. In that case we use mean, without (1/|e|) we would use sum.
    degree_edges = scatter(degree_vertices[V], E, dim=0, reduce=args.first_aggregate)
    # this is what goes into the UniGCN/UniGCNII formula
    # for d_i and d_e
    # x_i= 1/√d_i sum 1/√d_e Wh_e,
    degree_edges: torch.Tensor = degree_edges.pow(-0.5)
    degree_vertices: torch.Tensor = degree_vertices.pow(-0.5)
    degree_vertices[
        degree_vertices.isinf()
    ] = 1  # when not added self-loop, some nodes might not be connected with any edge

    V, E = V.to(device), E.to(device)
    args.degree_vertices = degree_vertices.to(device)
    args.degree_edges = degree_edges.to(device)
    args.dege2 = dege2.pow(-1.0).to(device)

    # nfeat: the dimension of the features
    # nclass: the number of classes in the labels
    nfeat: int
    nclass: int
    nfeat, nclass = X.shape[1], len(Y.unique())
    nlayer: int = args.nlayer
    nhid: int = args.nhid
    nhead: int = args.nhead

    # UniGNN and optimiser
    if args.model_name == "UniGCNII":
        model = UniGCNII(args, nfeat, nhid, nclass, nlayer, nhead)
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
        # include UniGCN, UniGAT, UniGIN, UniSAGE
        # print(f"args \n {args}")
        # print(type(args))
        # print(f"V \n {V}")
        # print(type(V))
        model = UniGNN(args, nfeat, nhid, nclass, nlayer, nhead)
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
