"""Architecture of UniGCN.

File taken from https://github.com/RaphaelPellegrin/UniGNN/tree/master"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter

from unignn_architectures.calculate_vertex_edges import calculate_v_e

# code from UniGCN paper
# https://github.com/RaphaelPellegrin/UniGNN/tree/master


# NOTE: can not tell which implementation is better statistically

# Check if CUDA is available and move tensors to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.heads,
        )

    def forward(
        self,
        X: torch.Tensor,
        vertex: torch.Tensor,
        edges: torch.Tensor,
        verbose: bool = False,
        hypergraph_classification: bool = False,
        dege: None | list = None,
        degv: None | list = None,
    ) -> torch.Tensor:
        device = X.device
        vertex = vertex.to(device)
        edges = edges.to(device)

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
        self,
        args,
        in_channels,
        out_channels,
        heads=8,
        dropout=0.0,
        negative_slope=0.2,
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
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.heads,
        )

    def forward(
        self,
        X: torch.Tensor,
        vertex: torch.Tensor,
        edges: torch.Tensor,
        verbose: bool = False,
        hypergraph_classification: bool = False,
        dege: None | list = None,
        degv: None | list = None,
    ) -> torch.Tensor:
        N = X.shape[0]
        X = X.to(device)
        vertex = vertex.to(device)
        edges = edges.to(device)
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
            in_features=in_channels,
            out_features=heads * out_channels,
            bias=False,
        )
        self.heads: int = heads
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.negative_slope: float = negative_slope
        self.dropout: float = dropout
        self.args = args

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.heads,
        )

    def forward(
        self,
        X: torch.Tensor,  # Node feature matrix (N, C) where N is the number of nodes and C is feature dimension
        vertex: torch.Tensor,
        edges: torch.Tensor,
        verbose: bool = False,
        hypergraph_classification: bool = False,
        dege: None | list = None,
        degv: None | list = None,
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
                or hg-level-classification (new, in which case need to pass deges, degvs, dege2s)
            deges:
                the degree of edges
            degvs:
                the degree of vertices

        Returns:
            x tilde i from UniGCN.
        """
        device = X.device
        vertex = vertex.to(device)
        edges = edges.to(device)
        # print(f"hypergraph_classification is {hypergraph_classification}")
        if verbose:
            print(f"vertex is {vertex}")
            print(f"edges {edges}")
            print(f"X is \n {X}")
        if hypergraph_classification:
            assert dege is not None
            assert degv is not None
            dege = torch.tensor(dege, dtype=torch.float32, device=device)
            degv = torch.tensor(degv, dtype=torch.float32, device=device)
        N: int = X.shape[0]  # the number of nodes
        if verbose:
            print(f"N is {N}")
        # combined degree of Edges
        if hypergraph_classification:
            pass
        else:
            if not hasattr(self.args, "dege"):
                available_attrs = [
                    attr for attr in dir(self.args) if not attr.startswith("_")
                ]
                raise AttributeError(
                    f"args must have dege attribute for non-hypergraph classification.\n"
                    f"Available attributes are: {available_attrs}"
                )
            if self.args.dege is None:
                raise ValueError(
                    f"args.dege cannot be None for non-hypergraph classification.\n"
                    f"args.dege type: {type(self.args.dege)}\n"
                    f"args.dege value: {self.args.dege}"
                )
            dege = torch.tensor(self.args.dege, device=device)
            degv = torch.tensor(self.args.degv, device=device)
        if verbose:
            print(f"dege is {dege}")
        # degree of vertices
        # TODO: Modify this so that it is a dictionary (ie self.args.degv is a dictionary, with the
        # first hg having it's first degv saved, etc)
        if hypergraph_classification:
            pass
        else:
            assert hasattr(
                self.args, "degv"
            ), "args must have degv attribute for non-hypergraph classification"
            assert (
                self.args.degv is not None
            ), "args.degv cannot be None for non-hypergraph classification"
            degv = self.args.degv
        if verbose:
            print(f"degv is {degv}")

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
        Xe = Xe * dege

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
            Xv * degv
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
        self,
        args,
        in_channels,
        out_channels,
        heads=8,
        dropout=0.0,
        negative_slope=0.2,
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
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.heads,
        )

    def forward(self, X, vertex, edges):
        device = X.device
        vertex = vertex.to(device)
        edges = edges.to(device)
        N = X.shape[0]
        dege = self.args.dege
        degv = self.args.degv

        # v3: X -> AX -> norm -> AXW

        Xve = X[vertex]  # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate)  # [E, C]

        Xe = Xe * dege

        Xev = Xe[edges]  # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce="sum", dim_size=N)  # [N, C]

        Xv = Xv * degv

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
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.heads,
        )

    def reset_parameters(self) -> None:
        glorot(self.att_v)
        glorot(self.att_e)

    # what would be smarter here, instead of passing unused arguments
    # would be to have a if model_name == "UniGCN" or "UniGAT" etc
    # and then have the arguments not passed! TODO later.
    def forward(
        self,
        X: torch.Tensor,
        vertex: torch.Tensor,
        edges: torch.Tensor,
        verbose: bool = False,
        hypergraph_classification: bool = False,
        dege: None | list = None,
        degv: None | list = None,
    ) -> torch.Tensor:
        device = X.device
        vertex = vertex.to(device)
        edges = edges.to(device)
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
                Conv(
                    args,
                    nhid * nhead,
                    nhid,
                    heads=nhead,
                    dropout=args.attn_drop,
                )
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
        verbose: bool = False,
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
            x_orig = X.clone()  # Create copy of original X
            X = conv(X, V, E)
            X = self.act(X)
            X = self.dropout(X)
            X = X.to(device)
            x_orig = x_orig.to(device)

            if self.args.do_transformer:
                if self.args.transformer_version == "v1":
                    # VERSION 1: Simple self-attention mechanism
                    # Uses a single scaled dot-product attention layer without positional encoding
                    # Advantages: Lightweight, faster training
                    # Disadvantages: Less expressive than full transformer

                    feature_dim = X.shape[-1]

                    # Handle dimension mismatch with projection if needed
                    # TODO: if we remove MP, could do X_without_encoding.shape[-1] and project there
                    if x_orig.shape[-1] != feature_dim:
                        if verbose:
                            print(f"feature_dim is {feature_dim}")
                            print(f"x_orig.shape is {x_orig.shape}")
                        projection = nn.Linear(
                            x_orig.shape[-1], feature_dim, device=X.device
                        )
                        x_orig = projection(x_orig)

                    # Add batch dimension for attention computation
                    x_transformer = x_orig.unsqueeze(0)  # [1, N, features]

                    # Apply basic self-attention using PyTorch's built-in function
                    # This is equivalent to a single attention head without the feed-forward network
                    x_transformer = F.scaled_dot_product_attention(
                        query=x_transformer,
                        key=x_transformer,
                        value=x_transformer,
                        dropout_p=self.dropout.p if self.training else 0.0,
                    )
                    x_transformer = x_transformer.to(device)

                    # Add residual connection
                    # TODO: remove MP: X = x_transformer
                    X = X + x_transformer.squeeze(0)

                elif self.args.transformer_version == "v2":
                    # VERSION 2: Full Transformer Encoder Architecture
                    # Uses complete transformer blocks with:
                    # - Multi-head attention
                    # - Feed-forward neural networks
                    # - Layer normalization
                    # - Residual connections
                    # Advantages: More expressive, better at capturing complex relationships
                    # Disadvantages: More parameters, slower training

                    feature_dim = X.shape[-1]

                    # Create or update full transformer architecture
                    if not hasattr(self, "transformer"):
                        # Create transformer encoder layer with:
                        # - Multi-head attention (nhead heads)
                        # - 4x larger feed-forward network
                        # - Dropout for regularization
                        self.transformer_layers = nn.TransformerEncoderLayer(
                            d_model=feature_dim,
                            nhead=self.args.nhead,
                            dim_feedforward=4
                            * feature_dim,  # Standard transformer uses 4x
                            dropout=self.args.dropout,
                            batch_first=True,
                        ).to(device)

                        # Stack multiple transformer layers
                        self.transformer = nn.TransformerEncoder(
                            self.transformer_layers,
                            num_layers=self.args.transformer_depth,
                        ).to(device)

                    # Handle dimension mismatch with projection if needed
                    if x_orig.shape[-1] != feature_dim:
                        if verbose:
                            print(f"feature_dim is {feature_dim}")
                            print(f"x_orig.shape[-1] is {x_orig.shape}")
                        projection = nn.Linear(
                            x_orig.shape[-1], feature_dim, device=X.device
                        ).to(device)
                        x_orig = projection(x_orig)

                    # Add batch dimension for transformer
                    x_transformer = x_orig.unsqueeze(0).to(device)
                    X = X.to(device)

                    # Apply full transformer encoding
                    x_transformer = self.transformer(x_transformer)

                    # Add residual connection
                    X = X + x_transformer.squeeze(0)

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
            What would be smarter would be to add deges and degvs to
            the dictionaries directly. Only need to compute it onnce, no need to pass it around:
            TODO later.

        Returns:
            List of prediction tensors (one per hypergraph)
        """
        list_preds: list[torch.Tensor] = []

        for dico in list_hypergraphs:
            # Get features and ensure they're float32
            X = torch.tensor(dico["features"], dtype=torch.float32)
            G = dico["hypergraph"]

            # Get or calculate degrees
            if "dege" in dico and "degv" in dico:
                dege = torch.tensor(dico["dege"], dtype=torch.float32)
                degv = torch.tensor(dico["degv"], dtype=torch.float32)
                # Convert vertex/edge indices from G
                V, E = [], []
                for edge_idx, (_, nodes) in enumerate(G.items()):
                    V.extend(nodes)
                    E.extend([edge_idx] * len(nodes))
                V = torch.tensor(V, dtype=torch.long)
                E = torch.tensor(E, dtype=torch.long)
            else:
                V, E, dege, degv, _ = calculate_v_e(X, G, self.args)
                dege = dege.float()
                degv = degv.float()

            # Forward pass through the network
            X = self.input_drop(X)
            for conv in self.convs:
                x_orig = X.clone()  # Store original features for transformer
                X = conv(
                    X=X,
                    vertex=V,
                    edges=E,
                    hypergraph_classification=True,
                    dege=dege,
                    degv=degv,
                )
                X = self.act(X)
                X = self.dropout(X)

                if self.args.do_transformer:
                    if self.args.transformer_version == "v1":
                        # VERSION 1: Simple self-attention mechanism
                        # Uses a single scaled dot-product attention layer without positional encoding
                        # Advantages: Lightweight, faster training
                        # Disadvantages: Less expressive than full transformer

                        feature_dim = X.shape[-1]

                        # Handle dimension mismatch with projection if needed
                        if x_orig.shape[-1] != feature_dim:
                            projection = nn.Linear(
                                x_orig.shape[-1], feature_dim, device=X.device
                            )
                            x_orig = projection(x_orig)

                        # Add batch dimension for attention computation
                        x_transformer = x_orig.unsqueeze(0)  # [1, N, features]

                        # Apply basic self-attention using PyTorch's built-in function
                        # This is equivalent to a single attention head without the feed-forward network
                        x_transformer = F.scaled_dot_product_attention(
                            query=x_transformer,
                            key=x_transformer,
                            value=x_transformer,
                            dropout_p=self.dropout.p if self.training else 0.0,
                        )

                        # Add residual connection
                        X = X + x_transformer.squeeze(0)

                    elif self.args.transformer_version == "v2":
                        # VERSION 2: Full Transformer Encoder Architecture
                        # Uses complete transformer blocks with:
                        # - Multi-head attention
                        # - Feed-forward neural networks
                        # - Layer normalization
                        # - Residual connections
                        # Advantages: More expressive, better at capturing complex relationships
                        # Disadvantages: More parameters, slower training

                        feature_dim = X.shape[-1]

                        # Create or update full transformer architecture
                        if not hasattr(self, "transformer"):
                            # Create transformer encoder layer with:
                            # - Multi-head attention (nhead heads)
                            # - 4x larger feed-forward network
                            # - Dropout for regularization
                            self.transformer_layers = nn.TransformerEncoderLayer(
                                d_model=feature_dim,
                                nhead=self.args.nhead,
                                dim_feedforward=4
                                * feature_dim,  # Standard transformer uses 4x
                                dropout=self.args.dropout,
                                batch_first=True,
                            ).to(device)

                            # Stack multiple transformer layers
                            self.transformer = nn.TransformerEncoder(
                                self.transformer_layers,
                                num_layers=self.args.transformer_depth,
                            ).to(device)

                        # Handle dimension mismatch with projection if needed
                        if x_orig.shape[-1] != feature_dim:
                            projection = nn.Linear(
                                x_orig.shape[-1], feature_dim, device=X.device
                            ).to(device)
                            x_orig = projection(x_orig)

                        # Add batch dimension for transformer
                        x_transformer = x_orig.unsqueeze(0).to(device)

                        # Apply full transformer encoding
                        x_transformer = self.transformer(x_transformer)

                        # Add residual connection
                        X = X + x_transformer.squeeze(0)

            # Output layer
            X = self.conv_out(
                X=X,
                vertex=V,
                edges=E,
                hypergraph_classification=True,
                dege=dege,
                degv=degv,
            )

            # Global pooling and prediction
            X_pooled = torch.mean(X, dim=0, keepdim=True)
            output = F.log_softmax(X_pooled, dim=1)
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
        dege = self.args.dege
        degv = self.args.degv

        Xve = X[vertex]  # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate)  # [E, C]

        Xe = Xe * dege

        Xev = Xe[edges]  # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce="sum", dim_size=N)  # [N, C]

        Xv = Xv * degv

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
    ) -> None:
        """UniGNNII

        Args:
            args:
                global args
            nfeat:
                dimension of features
            nhid:
                dimension of hidden features, note that actually it's #nhid x #nhead
            nclass:
                number of classes
            nlayer:
                number of hidden layers
            nhead:
                number of conv heads
        """
        super().__init__()
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

    def forward(
        self, x: torch.Tensor, V: torch.Tensor, E: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of UniGCNII

        Args:
            x: Input features
            V:
            E:
        """
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

    def forward_hypergraph_classification(
        self,
        list_hypergraphs: list,
    ) -> list[torch.Tensor]:
        """UniGCNII for hypergraph classification.

        Args:
            list_hypergraphs: List of dictionaries containing hypergraph data
                Each dict must have:
                    - 'features': node features tensor
                    - 'hypergraph': dictionary of edge lists
                    - Optional: 'dege', 'degv' (pre-computed degrees)

        Returns:
            List of prediction tensors (one per hypergraph)
        """
        list_preds: list[torch.Tensor] = []
        lamda, alpha = 0.5, 0.1  # Same hyperparameters as in forward()

        for dico in list_hypergraphs:
            # Get features and ensure they're float32
            X = torch.tensor(dico["features"], dtype=torch.float32)
            G = dico["hypergraph"]

            # Get or calculate degrees and indices
            if "dege" in dico and "degv" in dico:
                dege = torch.tensor(dico["dege"], dtype=torch.float32)
                degv = torch.tensor(dico["degv"], dtype=torch.float32)
                # Convert vertex/edge indices from G
                V, E = [], []
                for edge_idx, (_, nodes) in enumerate(G.items()):
                    V.extend(nodes)
                    E.extend([edge_idx] * len(nodes))
                V = torch.tensor(V, dtype=torch.long)
                E = torch.tensor(E, dtype=torch.long)
            else:
                V, E, dege, degv, _ = calculate_v_e(X, G, self.args)
                dege = dege.float()
                degv = degv.float()

            # Initial transformation
            X = self.dropout(X)
            X = F.relu(self.convs[0](X))
            X0 = X  # Store for skip connections

            # Convolution layers
            for i, conv in enumerate(self.convs[1:-1]):
                X = self.dropout(X)
                beta = math.log(lamda / (i + 1) + 1)
                X = F.relu(conv(X, V, E, alpha, beta, X0))

            # Final layer
            X = self.dropout(X)
            X = self.convs[-1](X)

            # Global pooling and prediction
            X_pooled = torch.mean(X, dim=0, keepdim=True)
            output = F.log_softmax(X_pooled, dim=1)
            list_preds.append(output)

        return list_preds
