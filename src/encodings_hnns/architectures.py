"""Architectures from UniGNN repo"""

import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch_geometric.utils import softmax
from torch_scatter import scatter

# files copied from UniGNN/models

# Gonna add comments, might make our own modifications


##########################################################################
############################### HyperGCN #################################
##########################################################################


class HyperGCN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, nlayer, V, E, X):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(HyperGCN, self).__init__()
        d, l, c = nfeat, nlayer, nclass
        cuda = True
        args.mediators = True

        h = [d]
        for i in range(l - 1):
            power = l - i + 2
            if args.dataset == "citeseer":
                power = l - i + 4
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
                for i in range(l)
            ]
        )
        self.do, self.l = args.dropout, args.nlayer
        self.structure, self.m = structure, args.mediators

    def forward(self, H):
        """
        an l-layer GCN
        """
        do, l, m = self.do, self.l, self.m

        for i, hidden in enumerate(self.layers):
            H = F.relu(hidden(self.structure, H, m))
            if i < l - 1:
                V = H
                H = F.dropout(H, do, training=self.training)

        return F.log_softmax(H, dim=1)


class HyperGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, a, b, reapproximate=True, cuda=True):
        super(HyperGraphConvolution, self).__init__()
        self.a, self.b = a, b
        self.reapproximate, self.cuda = reapproximate, cuda

        self.W = Parameter(torch.FloatTensor(a, b))
        self.bias = Parameter(torch.FloatTensor(b))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, structure, H, m=True):
        W, b = self.W, self.bias
        HW = torch.mm(H, W)

        if self.reapproximate:
            n, X = H.shape[0], HW.cpu().detach().numpy()
            A = Laplacian(n, structure, X, m)
        else:
            A = structure

        if self.cuda:
            A = A.cuda()
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
        ctx.save_for_backward(M1, M2)
        return torch.mm(M1, M2)

    @staticmethod
    def backward(ctx, g):
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
                        [[Se, mediator], [Ie, mediator], [mediator, Se], [mediator, Ie]]
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


def symnormalise(M):
    """
    symmetrically normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1/2} M D^{-1/2}
    where D is the diagonal node-degree matrix
    """

    d = np.array(M.sum(1))

    dhi = np.power(d, -1 / 2).flatten()
    dhi[np.isinf(dhi)] = 0.0
    DHI = sp.diags(dhi)  # D half inverse i.e. D^{-1/2}

    return (DHI.dot(M)).dot(DHI)


def ssm2tst(M):
    """
    converts a scipy sparse matrix (ssm) to a torch sparse tensor (tst)

    arguments:
    M: scipy sparse matrix

    returns:
    a torch sparse tensor of M
    """

    M = M.tocoo().astype(np.float32)

    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


#################################################################
##################### UniGNN ####################################


# NOTE: can not tell which implementation is better statistically


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def normalize_l2(X):
    """Row-normalize  matrix"""
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.0
    X = X * scale
    return X


# v1: X -> XW -> AXW -> norm
class UniSAGEConv(nn.Module):

    def __init__(
        self, args, in_channels, out_channels, heads=8, dropout=0.0, negative_slope=0.2
    ):
        super().__init__()
        # TODO: bias?
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)

        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.args = args

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )

    def forward(self, X, vertex, edges):
        N = X.shape[0]

        # X0 = X # NOTE: reserved for skip connection

        X = self.W(X)

        Xve = X[vertex]  # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate)  # [E, C]

        Xev = Xe[edges]  # [nnz, C]
        Xv = scatter(
            Xev, vertex, dim=0, reduce=self.args.second_aggregate, dim_size=N
        )  # [N, C]
        X = X + Xv

        if self.args.use_norm:
            X = normalize_l2(X)

        # NOTE: concat heads or mean heads?
        # NOTE: normalize here?
        # NOTE: skip concat here?

        return X


# v1: X -> XW -> AXW -> norm
class UniGINConv(nn.Module):

    def __init__(
        self, args, in_channels, out_channels, heads=8, dropout=0.0, negative_slope=0.2
    ):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)

        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.eps = nn.Parameter(torch.Tensor([0.0]))
        self.args = args

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        # X0 = X # NOTE: reserved for skip connection

        # v1: X -> XW -> AXW -> norm
        X = self.W(X)

        Xve = X[vertex]  # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate)  # [E, C]

        Xev = Xe[edges]  # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce="sum", dim_size=N)  # [N, C]
        X = (1 + self.eps) * X + Xv

        if self.args.use_norm:
            X = normalize_l2(X)

        # NOTE: concat heads or mean heads?
        # NOTE: normalize here?
        # NOTE: skip concat here?

        return X


# v1: X -> XW -> AXW -> norm
class UniGCNConv(nn.Module):

    def __init__(
        self, args, in_channels, out_channels, heads=8, dropout=0.0, negative_slope=0.2
    ):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.args = args

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        degE = self.args.degE
        degV = self.args.degV

        # v1: X -> XW -> AXW -> norm

        X = self.W(X)

        Xve = X[vertex]  # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate)  # [E, C]

        Xe = Xe * degE

        Xev = Xe[edges]  # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce="sum", dim_size=N)  # [N, C]

        Xv = Xv * degV

        X = Xv

        if self.args.use_norm:
            X = normalize_l2(X)

        # NOTE: skip concat here?

        return X


# v2: X -> AX -> norm -> AXW
class UniGCNConv2(nn.Module):

    def __init__(
        self, args, in_channels, out_channels, heads=8, dropout=0.0, negative_slope=0.2
    ):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=True)
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.args = args

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        degE = self.args.degE
        degV = self.args.degV

        # v3: X -> AX -> norm -> AXW

        Xve = X[vertex]  # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate)  # [E, C]

        Xe = Xe * degE

        Xev = Xe[edges]  # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce="sum", dim_size=N)  # [N, C]

        Xv = Xv * degV

        X = Xv

        if self.args.use_norm:
            X = normalize_l2(X)

        X = self.W(X)

        # NOTE: result might be slighly unstable
        # NOTE: skip concat here?

        return X


class UniGATConv(nn.Module):

    def __init__(
        self,
        args,
        in_channels,
        out_channels,
        heads=8,
        dropout=0.0,
        negative_slope=0.2,
        skip_sum=False,
    ):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)

        self.att_v = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_drop = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.skip_sum = skip_sum
        self.args = args
        self.reset_parameters()

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )

    def reset_parameters(self):
        glorot(self.att_v)
        glorot(self.att_e)

    def forward(self, X, vertex, edges):
        H, C, N = self.heads, self.out_channels, X.shape[0]

        # X0 = X # NOTE: reserved for skip connection

        X0 = self.W(X)
        X = X0.view(N, H, C)

        Xve = X[vertex]  # [nnz, H, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate)  # [E, H, C]

        alpha_e = (Xe * self.att_e).sum(-1)  # [E, H, 1]
        a_ev = alpha_e[edges]
        alpha = a_ev  # Recommed to use this
        alpha = self.leaky_relu(alpha)
        alpha = softmax(alpha, vertex, num_nodes=N)
        alpha = self.attn_drop(alpha)
        alpha = alpha.unsqueeze(-1)

        Xev = Xe[edges]  # [nnz, H, C]
        Xev = Xev * alpha
        Xv = scatter(Xev, vertex, dim=0, reduce="sum", dim_size=N)  # [N, H, C]
        X = Xv
        X = X.view(N, H * C)

        if self.args.use_norm:
            X = normalize_l2(X)

        if self.skip_sum:
            X = X + X0

        # NOTE: concat heads or mean heads?
        # NOTE: skip concat here?

        return X


__all_convs__ = {
    "UniGAT": UniGATConv,
    "UniGCN": UniGCNConv,
    "UniGCN2": UniGCNConv2,
    "UniGIN": UniGINConv,
    "UniSAGE": UniSAGEConv,
}


class UniGNN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, nlayer, nhead, V, E):
        """UniGNN

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        Conv = __all_convs__[args.model_name]
        self.conv_out = Conv(
            args, nhid * nhead, nclass, heads=1, dropout=args.attn_drop
        )
        self.convs = nn.ModuleList(
            [Conv(args, nfeat, nhid, heads=nhead, dropout=args.attn_drop)]
            + [
                Conv(args, nhid * nhead, nhid, heads=nhead, dropout=args.attn_drop)
                for _ in range(nlayer - 2)
            ]
        )
        self.V = V
        self.E = E
        act = {"relu": nn.ReLU(), "prelu": nn.PReLU()}
        self.act = act[args.activation]
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, X):
        V, E = self.V, self.E

        X = self.input_drop(X)
        for conv in self.convs:
            X = conv(X, V, E)
            X = self.act(X)
            X = self.dropout(X)

        X = self.conv_out(X, V, E)
        return F.log_softmax(X, dim=1)


class UniGCNIIConv(nn.Module):
    def __init__(self, args, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.args = args

    def forward(self, X, vertex, edges, alpha, beta, X0):
        N = X.shape[0]
        degE = self.args.degE
        degV = self.args.degV

        Xve = X[vertex]  # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate)  # [E, C]

        Xe = Xe * degE

        Xev = Xe[edges]  # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce="sum", dim_size=N)  # [N, C]

        Xv = Xv * degV

        X = Xv

        if self.args.use_norm:
            X = normalize_l2(X)

        Xi = (1 - alpha) * X + alpha * X0
        X = (1 - beta) * Xi + beta * self.W(Xi)

        return X


class UniGCNII(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, nlayer, nhead, V, E):
        """UniGNNII

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        self.V = V
        self.E = E
        nhid = nhid * nhead
        act = {"relu": nn.ReLU(), "prelu": nn.PReLU()}
        self.act = act[args.activation]
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(nfeat, nhid))
        for _ in range(nlayer):
            self.convs.append(UniGCNIIConv(args, nhid, nhid))
        self.convs.append(torch.nn.Linear(nhid, nclass))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters()) + list(
            self.convs[-1:].parameters()
        )
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        V, E = self.V, self.E
        lamda, alpha = 0.5, 0.1
        x = self.dropout(x)
        x = F.relu(self.convs[0](x))
        x0 = x
        for i, con in enumerate(self.convs[1:-1]):
            x = self.dropout(x)
            beta = math.log(lamda / (i + 1) + 1)
            x = F.relu(con(x, V, E, alpha, beta, x0))
        x = self.dropout(x)
        x = self.convs[-1](x)
        return F.log_softmax(x, dim=1)
