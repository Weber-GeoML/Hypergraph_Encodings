"""File taken from https://github.com/RaphaelPellegrin/UniGNN/tree/master"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter

from uniGCN.calculate_vertex_edges import calculate_V_E

# code from UniGCN paper
# https://github.com/RaphaelPellegrin/UniGNN/tree/master


# NOTE: can not tell which implementation is better statistically


def glorot(tensor):
    """TODO

    Args:
        tensor:
            TODO

    """
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def normalize_l2(X):
    """Row-normalize  matrix

    Args:
        X:
            TODO

    Returns:
        X:
            TODO
    """
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.0
    X = X * scale
    return X


# v1: X -> XW -> AXW -> norm
class UniSAGEConv(nn.Module):

    def __init__(
        self,
        args,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
    ) -> None:
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
        """TODO

        Args:
            X:
                TODO
            vertex:
                TODO
            edges:
                TODO

        Returns:
            TODO

        """
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
        """TODO

        Args:
            X:
                TODO
            vertex:
                TODO
            edges:
                TODO

        """
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
        self,
        args,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
    ) -> None:
        """

        Args:
            args,
            in_channels:
                dimension of input (in our case it will be feature dimension)
            out_channels:
                dimension of output
            heads:
                number of conv heads
            dropout:
                TODO
            negative_slope:
                TODO

        """
        super().__init__()
        # in_features: size of each input sample
        # out_features: size of each output sample
        self.W = nn.Linear(
            in_features=in_channels, out_features=heads * out_channels, bias=False
        )
        self.heads: int = heads
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.negative_slope: float = negative_slope
        self.dropout: float = dropout
        self.args = args

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )

    def forward(
        self,
        X: torch.Tensor,  # Node feature matrix (N, C) where N is the number of nodes and C is feature dimension
        vertex: torch.Tensor,
        edges: torch.Tensor,
        verbose: bool = False,
        hypergraph_classification: bool = False,
        degE: None | list = None,
        degV: None | list = None,
    ) -> torch.Tensor:
        """Performs 1/srqrt(di) * [sum_e 1/sqrt(d_e)Wh_e], ie x tilde i

        This is the formula for UniGCN from the paper. What it does
        here is that it does it all at once.

        Args:
            X:
                the tensor of features.
                Size is number of nodes times dim of features ie N by C
            vertex:
                a tensor of vertices.
                this will be of the form eg [1, 0, 1]
                with edges [0, 1, 1]
                meaning vertex 1 is in edge 0
                and vertex 0 is in edge 1
                and vertex 1 is in edge 1
            edges:
                see above
            hypergraph_classification:
                wether we do node-level, within hg classification,
                or hg-level-classification (new, in which case need to pass degEs, degVs, degE2s)
            degEs:
                the degree of edges
            degVs:
                the degree of vertices

        Returns:
            x tilde i from UniGCN.
        """
        # print(f"hypergraph_classification is {hypergraph_classification}")
        if verbose:
            print(f"vertex is {vertex}")
            print(f"edges {edges}")
            print(f"X is \n {X}")
        if hypergraph_classification:
            assert degE is not None
            assert degV is not None
        N: int = X.shape[0]  # the number of nodes
        if verbose:
            print(f"N is {N}")
        # combined degree of Edges
        if hypergraph_classification:
            pass
        else:
            degE = self.args.degE
        if verbose:
            print(f"degE is {degE}")
        # degree of vertices
        # TODO: Modify this so that it is a dictionary (ie self.args.degV is a dictionary, with the
        # first hg having it's first degV saved, etc)
        if hypergraph_classification:
            pass
        else:
            degV = self.args.degV
        if verbose:
            print(f"degV is {degV}")

        # v1: X -> XW -> AXW -> norm

        # first step: aggregate node attribute at the edge level.
        # multiply by W
        X = self.W(X)

        # select the relevant features for the vertices
        # ie this grabs the relevant rows
        Xve = X[vertex]  # [number of elem in vertex, C]
        # aggregation at the edge level using node features.
        # called h_e
        Xe: torch.Tensor = scatter(
            Xve, edges, dim=0, reduce=self.args.first_aggregate
        )  # [E, C]

        # 1/sqrt(d_e)Wh_e
        Xe = Xe * degE

        # this return the row ofs Xev for edges
        # ie only the features h_e for edges
        Xev = Xe[edges]  # [nnz, C]
        # [sum_e 1/sqrt(d_e)Wh_e]
        # aggregate at the node level using edge features
        Xv: torch.Tensor = scatter(
            Xev, vertex, dim=0, reduce="sum", dim_size=N
        )  # [N, C]

        # 1/srqrt(di) * [sum_e 1/sqrt(d_e)Wh_e]
        # where [sum_e 1/sqrt(d_e)Wh_2] is Xv
        Xv = (
            Xv * degV
        )  # Scale node features by the reciprocal of the square root of vertex degrees

        X = Xv

        # Optionally apply L2 normalization to the node features
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
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        skip_sum: bool = False,
    ) -> None:
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

    def reset_parameters(self) -> None:
        glorot(self.att_v)
        glorot(self.att_e)

    def forward(self, X, vertex, edges):
        """TODO

        Args:
            X:
                TODO
            vertex:
                TODO
            edges:
                TODO
        """
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


__all_convs__: dict = {
    "UniGAT": UniGATConv,
    "UniGCN": UniGCNConv,
    "UniGCN2": UniGCNConv2,
    "UniGIN": UniGINConv,
    "UniSAGE": UniSAGEConv,
}


class UniGNN(nn.Module):
    def __init__(
        self,
        args,
        nfeat: int,  # dimension of features
        nhid: int,
        nclass: int,  # number of classes
        nlayer: int,  # depth of the neural network.
        nhead: int,  # number of heads
    ) -> None:
        """UniGNN

        Args:
            args:
                global args
            nfeat:
                dimension of features
            nhid:
                dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass:
                number of classes
            nlayer:
                number of hidden layers
            nhead:
                number of conv heads.
        """
        super().__init__()
        Conv = __all_convs__[args.model_name]  # this is just UniGCNConv
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
        act = {"relu": nn.ReLU(), "prelu": nn.PReLU()}
        self.act = act[args.activation]
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)
        self.args = args

    def forward(
        self,
        X: torch.Tensor,
        V: torch.Tensor,
        E: torch.Tensor,
    ) -> torch.Tensor:
        """TODO:

        Args:
            X:
                the features.
                Has shape number of nodes time size of features.

        Returns:
            a tensor/vector that has gone
            through a softmax.
        """

        X = self.input_drop(X)
        for conv in self.convs:  # note that we loop for as many layers as specified
            # TODO: create a copy of the original X
            X = conv(X, V, E)
            X = self.act(X)
            X = self.dropout(X)
            # TODO: transformer
            # TODO: combine original X and transfomer(X)

        X = self.conv_out(X, V, E)
        return F.log_softmax(X, dim=1)

    def forward_hypergraph_classification(
        self,
        list_hypergraphs: list,
    ) -> list[torch.Tensor | float]:
        """UniGCN for hypergraph classification.

        Performs the same as regular UniGCN, but then aggregates
        all outputs (node level) to have just one prediction at the hypergraph
        level.

        Args:
            list_hypergraphs:
                the list of dicts (that contains hg, features, labels etc)
            degEs:
                the list of the relevant degEs
            degVs:
                the list of the relevant degVs
            What would be smarter would be to add degEs and degVs to
            the dictionaries directly. Only need to compute it onnce, no need to pass it around:
            TODO later.

        Returns:
            a tensor/vector that has gone
            through a softmax.
        """
        list_preds: list[float] = []
        for idx, dico in enumerate(list_hypergraphs):
            X: torch.Tensor
            G: dict
            X = dico["features"]
            G = dico["hypergraph"]
            V, E, degE, degV, degE2 = calculate_V_E(X, G, self.args)
            # Do I give V, E here. DO I compute before?
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X)
            X = self.input_drop(X)
            for conv in self.convs:  # note that we loop for as many layers as specified
                X = conv(
                    X=X,
                    vertex=V,
                    edges=E,
                    hypergraph_classification=True,
                    degE=degE,
                    degV=degV,
                )
                X = self.act(X)
                X = self.dropout(X)

            X = self.conv_out(
                X=X,
                vertex=V,
                edges=E,
                hypergraph_classification=True,
                degE=degE,
                degV=degV,
            )
            # Aggregating using mean (you can also use sum or other methods) or sum
            X_aggregated = torch.mean(X, dim=0).unsqueeze(0)  # Shape (1, 2)
            output = F.log_softmax(X_aggregated, dim=1)
            list_preds.append(output)

        return list_preds


class UniGCNIIConv(nn.Module):
    def __init__(self, args, in_features: int, out_features: int) -> None:
        super().__init__()
        # this is just a matrix
        self.W: torch.Tensor = nn.Linear(in_features, out_features, bias=False)
        self.args = args

    def forward(
        self,
        X: torch.Tensor,
        vertex: torch.Tensor,
        edges: torch.Tensor,
        alpha: float,
        beta: float,
        X0: torch.Tensor,
    ) -> torch.Tensor:
        """TODO

        Args:
            X:
                the torch-tensor of features.
            vertex:
                TODO
            edges:
                TODO
            alpha:
                a hyper-parameter that goes into UniGCNII.
            beta:
                a hyper-parameter that goes into UniGCNII
            X0:
                the values of x_i at time 0.
        """
        N: int = X.shape[0]
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

        # alpha is a hyper-parameter. XO is the original feature
        # this is the whole trick of UniGCNII to avoid over-smoothing
        # use X0
        Xi = (1 - alpha) * X + alpha * X0
        # this is the UniGCNII x tile i
        # beta is a hyperparameter
        X = (1 - beta) * Xi + beta * self.W(Xi)

        return X


class UniGCNII(nn.Module):
    def __init__(
        self,
        args,
        nfeat: int,
        nhid: int,
        nclass: int,
        nlayer: int,
        nhead: int,
        V: torch.LongTensor,
        E: torch.LongTensor,
    ) -> None:
        """UniGNNII

        Args:
            args:
                global args
            nfeat:
                dimension of features
            nhid:
                dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass:
                number of classes
            nlayer:
                number of hidden layers
            nhead:
                number of conv heads
            V:
                V is the row index for the sparse incident matrix H, |V| x |E|
            E:
                E is the col index for the sparse incident matrix H, |V| x |E|
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
        """TODO

        Args:
            x:
                TODO
        """
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
