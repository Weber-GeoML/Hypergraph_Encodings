import scipy.sparse as sp
import torch
import torch_sparse
from torch_scatter import scatter

# Check if CUDA is available and move tensors to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_V_E(X: torch.Tensor, G: dict, args) -> tuple:
    """Calculates V and E

    Arguments:
        X:
            the features
        G:
            the graph

    Returns:
        a tuple with V, E, degE, degV, degE2.
            V: the row indices of the non-zero elements in H
            E: the column indices of the non-zero elements in H
            degE: the degree of each edge
            degV: the degree of each vertex
            degE2: the degree of each edge weighted by the degree of the vertices

    """

    G = G.copy()

    if args.add_self_loop:
        # Create a set of all node indices from 0 to number_of_nodes-1
        # This set will track which nodes still need self-loops
        Vs = set(range(X.shape[0]))

        # Iterate through all edges in the hypergraph
        for edge, nodes in G.items():
            # Check if this edge is already a self-loop:
            # - It contains exactly one node (len(nodes) == 1)
            # - That node is in our tracking set (nodes[0] in Vs)
            if len(nodes) == 1 and nodes[0] in Vs:
                # If we find an existing self-loop for a node,
                # remove that node from our tracking set
                # because it doesn't need another self-loop
                Vs.remove(nodes[0])

        # After checking existing edges, Vs now contains only
        # the nodes that don't have any self-loops yet
        # Add a new self-loop edge for each of these nodes
        for v in Vs:
            # Create a new edge named "self-loop-{node_index}"
            # containing only that node
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
    degV: torch.Tensor = (
        torch.from_numpy(H.sum(1)).view(-1, 1).float()
    )  # the degree of each vertices
    degE2: torch.Tensor = (
        torch.from_numpy(H.sum(0)).view(-1, 1).float()
    )  # the degree of each edge
    degE = scatter(degV[V], E, dim=0, reduce=args.first_aggregate)
    # this is what goes into the UniGCN/UniGCNII formula
    # for d_i and d_e
    # x_i= 1/√d_i sum 1/√d_e Wh_e,
    degE: torch.Tensor = degE.pow(-0.5)
    degV: torch.Tensor = degV.pow(-0.5)
    degV[
        degV.isinf()
    ] = 1  # when not added self-loop, some nodes might not be connected with any edge
    return V, E, degE, degV, degE2
