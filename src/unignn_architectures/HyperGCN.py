"""File taken from https://github.com/RaphaelPellegrin/UniGNN/tree/master"""

import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

# Code from uniGCN paper
# https://github.com/RaphaelPellegrin/UniGNN/tree/master

# Check if CUDA is available and move tensors to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HyperGCN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass: int, nlayer: int, V, E, X) -> None:
        """Initialize the HyperGCN model.

        Args:
            d:
                initial node-feature dimension
            h:
                number of hidden units
            c:
                number of classes
        """
        super(HyperGCN, self).__init__()
        d, layer_l, c = nfeat, nlayer, nclass
        cuda = False
        args.mediators = True

        h = [d]
        for i in range(layer_l - 1):
            power = layer_l - i + 2
            if args.dataset == "citeseer":
                power = layer_l - i + 4
            h.append(2**power)
        h.append(c)

        if args.fast:
            reapproximate = False
            structure = Laplacian(V, E, X, args.mediators)
        else:
            reapproximate = True
            structure = E

        self.layers = nn.ModuleList(
            [
                HyperGraphConvolution(h[i], h[i + 1], reapproximate, cuda)
                for i in range(layer_l)
            ]
        )
        self.do, self.l = args.dropout, args.nlayer
        self.structure, self.m = structure, args.mediators

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """An l-layer GCN.

        Args:
            H:
                TODO

        Returns:
            The output features.
        """
        do, layer_l, m = self.do, self.l, self.m

        for i, hidden in enumerate(self.layers):
            H = F.relu(hidden(self.structure, H, m))
            if i < layer_l - 1:
                H = F.dropout(H, do, training=self.training)

        return F.log_softmax(H, dim=1)


class HyperGraphConvolution(Module):
    """Simple GCN layer.

    Similar to https://arxiv.org/abs/1609.02907.
    """

    def __init__(self, a, b, reapproximate=True, cuda=False):
        """Initialize the HyperGraphConvolution layer.

        Args:
            a:
                TODO
            b:
                TODO
        """
        super(HyperGraphConvolution, self).__init__()
        self.a, self.b = a, b
        self.reapproximate, self.cuda = reapproximate, cuda

        self.W = Parameter(torch.FloatTensor(a, b))
        self.bias = Parameter(torch.FloatTensor(b))
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of the layer."""
        std = 1.0 / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, structure, H, m: int = True):
        """Forward pass of the layer.

        Args:
            structure:
                TODO
            H:
                TODO
            m:
                TODO
        """
        W, b = self.W, self.bias
        HW = torch.mm(H, W)

        if self.reapproximate:
            n, X = H.shape[0], HW.cpu().detach().numpy()
            A = Laplacian(n, structure, X, m)
        else:
            A = structure

        A = A.to(device)
        A = Variable(A)

        AHW = SparseMM.apply(A, HW)
        return AHW + b

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.a) + " -> " + str(self.b) + ")"


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    @staticmethod
    def forward(ctx, M1, M2):
        """Forward pass of the SparseMM layer.

        Args:
            ctx:
                TODO
            M1:
                TODO
        """
        ctx.save_for_backward(M1, M2)
        return torch.mm(M1, M2)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, g: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backward pass of the SparseMM layer.

        Args:
            g:
                The gradient.

        Returns:
            The gradient of the first matrix.
            The gradient of the second matrix.
        """
        M1, M2 = ctx.saved_tensors
        g1 = g2 = None

        if ctx.needs_input_grad[0]:
            g1 = torch.mm(g, M2.t())

        if ctx.needs_input_grad[1]:
            g2 = torch.mm(M1.t(), g)

        return g1, g2


def Laplacian(V, E, X, m):
    """
    approximates the E defined by the E Laplacian with/without mediators

    arguments:
    V: number of vertices
    E: dictionary of hyperedges (key: hyperedge, value: list/set of hypernodes)
    X: features on the vertices
    m: True gives Laplacian with mediators, while False gives without

    A: adjacency matrix of the graph approximation
    returns:
    updated data with 'graph' as a key and its value the approximated hypergraph
    """

    edges, weights = [], {}
    rv = np.random.rand(X.shape[1])

    for k in E.keys():
        hyperedge = list(E[k])

        p = np.dot(X[hyperedge], rv)  # projection onto a random vector rv
        s, i = np.argmax(p), np.argmin(p)
        Se, Ie = hyperedge[s], hyperedge[i]

        # two stars with mediators
        c = 2 * len(hyperedge) - 3  # normalisation constant
        if m:

            # connect the supremum (Se) with the infimum (Ie)
            edges.extend([[Se, Ie], [Ie, Se]])

            if (Se, Ie) not in weights:
                weights[(Se, Ie)] = 0
            weights[(Se, Ie)] += float(1 / c)

            if (Ie, Se) not in weights:
                weights[(Ie, Se)] = 0
            weights[(Ie, Se)] += float(1 / c)

            # connect the supremum (Se) and the infimum (Ie) with each mediator
            for mediator in hyperedge:
                if mediator != Se and mediator != Ie:
                    edges.extend(
                        [
                            [Se, mediator],
                            [Ie, mediator],
                            [mediator, Se],
                            [mediator, Ie],
                        ]
                    )
                    weights = update(Se, Ie, mediator, weights, c)
        else:
            edges.extend([[Se, Ie], [Ie, Se]])
            e = len(hyperedge)

            if (Se, Ie) not in weights:
                weights[(Se, Ie)] = 0
            weights[(Se, Ie)] += float(1 / e)

            if (Ie, Se) not in weights:
                weights[(Ie, Se)] = 0
            weights[(Ie, Se)] += float(1 / e)

    return adjacency(edges, weights, V)


def update(Se, Ie, mediator, weights, c):
    """
    updates the weight on {Se,mediator} and {Ie,mediator}
    """

    if (Se, mediator) not in weights:
        weights[(Se, mediator)] = 0
    weights[(Se, mediator)] += float(1 / c)

    if (Ie, mediator) not in weights:
        weights[(Ie, mediator)] = 0
    weights[(Ie, mediator)] += float(1 / c)

    if (mediator, Se) not in weights:
        weights[(mediator, Se)] = 0
    weights[(mediator, Se)] += float(1 / c)

    if (mediator, Ie) not in weights:
        weights[(mediator, Ie)] = 0
    weights[(mediator, Ie)] += float(1 / c)

    return weights


def adjacency(edges, weights, n):
    """
    computes an sparse adjacency matrix

    arguments:
    edges: list of pairs
    weights: dictionary of edge weights (key: tuple representing edge, value: weight on the edge)
    n: number of nodes

    returns: a scipy.sparse adjacency matrix with unit weight self loops for edges with the given weights
    """

    dictionary = {tuple(item): index for index, item in enumerate(edges)}
    edges = [list(itm) for itm in dictionary.keys()]
    organised = []

    for e in edges:
        i, j = e[0], e[1]
        w = weights[(i, j)]
        organised.append(w)

    edges, weights = np.array(edges), np.array(organised)
    adj = sp.coo_matrix(
        (weights, (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32
    )
    adj = adj + sp.eye(n)

    A = symnormalise(sp.csr_matrix(adj, dtype=np.float32))
    A = ssm2tst(A)
    return A


def symnormalise(M: sp.csr_matrix) -> sp.csr_matrix:
    """Symmetrically normalise sparse matrix.

    Args:
        M:
            scipy sparse matrix

    Returns:
        D^{-1/2} M D^{-1/2}
        where D is the diagonal node-degree matrix
    """

    d = np.array(M.sum(1))

    dhi = np.power(d, -1 / 2).flatten()
    dhi[np.isinf(dhi)] = 0.0
    DHI = sp.diags(dhi)  # D half inverse i.e. D^{-1/2}

    return (DHI.dot(M)).dot(DHI)


def ssm2tst(M: sp.csr_matrix) -> torch.sparse.FloatTensor:
    """Converts a scipy sparse matrix (ssm) to a torch sparse tensor (tst).

    Args:
        M:
            scipy sparse matrix

    Returns:
        a torch sparse tensor of M
    """

    M = M.tocoo().astype(np.float32)

    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)

    return torch.sparse.FloatTensor(indices, values, shape)
