"""We could import TopModelX and use their structure
The reason we are copying the files directly is because we want to be able to modify them ourslves.

I feel like it would make sense to keep in separate files like they did in their Topomodelx repo though.
That way it is easier to debug.
"""

import math
from typing import Literal

import numpy as np
import torch
from scipy.sparse import _csc
from torch.nn.parameter import Parameter


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    """Broadcasts `src` to the shape of `other`."""
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    return src.expand(other.size())


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    """Add all values from the `src` tensor into `out` at the indices."""
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)

    return out.scatter_add_(dim, index, src)


def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    """Add all values from the `src` tensor into `out` at the indices."""
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    """Compute the mean value of all values from the `src` tensor into `out`."""
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode="floor")
    return out


SCATTER_DICT = {"sum": scatter_sum, "mean": scatter_mean, "add": scatter_sum}


def scatter(scatter: str):
    """Return the scatter function."""
    if scatter not in SCATTER_DICT:
        raise ValueError(f"scatter must be string: {list(SCATTER_DICT.keys())}")

    return SCATTER_DICT[scatter]


class MessagePassing(torch.nn.Module):
    """Define message passing.

    Originally in topomodelx/base/message_passing

    This class defines message passing through a single neighborhood N,
    by decomposing it into 2 steps:

    1. 🟥 Create messages going from source cells to target cells through N.
    2. 🟧 Aggregate messages coming from different sources cells onto each target cell.

    This class should not be instantiated directly, but rather inherited
    through subclasses that effectively define a message passing function.

    This class does not have trainable weights, but its subclasses should
    define these weights.

    Parameters
    ----------
    aggr_func : Literal["sum", "mean", "add"], default="sum"
        Aggregation function to use.
    att : bool, default=False
        Whether to use attention.
    initialization : Literal["uniform", "xavier_uniform", "xavier_normal"], default="xavier_uniform"
        Initialization method for the weights of the layer.
    initialization_gain : float, default=1.414
        Gain for the weight initialization.

    References
    ----------
    .. [1] Hajij, Zamzmi, Papamarkou, Miolane, Guzmán-Sáenz, Ramamurthy, Birdal, Dey,
        Mukherjee, Samaga, Livesay, Walters, Rosen, Schaub.
        Topological deep learning: going beyond graph data (2023).
        https://arxiv.org/abs/2206.00606.

    .. [2] Papillon, Sanborn, Hajij, Miolane.
        Architectures of topological deep learning: a survey on topological neural networks (2023).
        https://arxiv.org/abs/2304.10031.
    """

    def __init__(
        self,
        aggr_func: Literal["sum", "mean", "add"] = "sum",
        att: bool = False,
        initialization: Literal[
            "uniform", "xavier_uniform", "xavier_normal"
        ] = "xavier_uniform",
        initialization_gain: float = 1.414,
    ) -> None:
        super().__init__()
        self.aggr_func = aggr_func
        self.att = att
        self.initialization = initialization
        self.initialization_gain = initialization_gain

    def reset_parameters(self):
        r"""Reset learnable parameters.

        Notes
        -----
        This function will be called by subclasses of MessagePassing that have trainable weights.
        """
        match self.initialization:
            case "uniform":
                if self.weight is not None:
                    stdv = 1.0 / math.sqrt(self.weight.size(1))
                    self.weight.data.uniform_(-stdv, stdv)
                if self.att:
                    stdv = 1.0 / math.sqrt(self.att_weight.size(1))
                    self.att_weight.data.uniform_(-stdv, stdv)
            case "xavier_uniform":
                if self.weight is not None:
                    torch.nn.init.xavier_uniform_(
                        self.weight, gain=self.initialization_gain
                    )
                if self.att:
                    torch.nn.init.xavier_uniform_(
                        self.att_weight.view(-1, 1), gain=self.initialization_gain
                    )
            case "xavier_normal":
                if self.weight is not None:
                    torch.nn.init.xavier_normal_(
                        self.weight, gain=self.initialization_gain
                    )
                if self.att:
                    torch.nn.init.xavier_normal_(
                        self.att_weight.view(-1, 1), gain=self.initialization_gain
                    )
            case _:
                raise ValueError(
                    f"Initialization {self.initialization} not recognized."
                )

    def message(self, x_source, x_target=None):
        """Construct message from source cells to target cells.

        🟥 This provides a default message function to the message passing scheme.

        Alternatively, users can subclass MessagePassing and overwrite
        the message method in order to replace it with their own message mechanism.

        Parameters
        ----------
        x_source : Tensor, shape = (..., n_source_cells, in_channels)
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        x_target : Tensor, shape = (..., n_target_cells, in_channels)
            Input features on target cells.
            Assumes that all target cells have the same rank s.
            Optional. If not provided, x_target is assumed to be x_source,
            i.e. source cells send messages to themselves.

        Returns
        -------
        torch.Tensor, shape = (..., n_source_cells, in_channels)
            Messages on source cells.
        """
        return x_source

    def attention(self, x_source, x_target=None):
        """Compute attention weights for messages.

        This provides a default attention function to the message-passing scheme.

        Alternatively, users can subclass MessagePassing and overwrite
        the attention method in order to replace it with their own attention mechanism.

        The implementation follows [1]_.

        Parameters
        ----------
        x_source : torch.Tensor, shape = (n_source_cells, in_channels)
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        x_target : torch.Tensor, shape = (n_target_cells, in_channels)
            Input features on source cells.
            Assumes that all source cells have the same rank r.

        Returns
        -------
        torch.Tensor, shape = (n_messages, 1)
            Attention weights: one scalar per message between a source and a target cell.
        """
        x_source_per_message = x_source[self.source_index_j]
        x_target_per_message = (
            x_source[self.target_index_i]
            if x_target is None
            else x_target[self.target_index_i]
        )

        x_source_target_per_message = torch.cat(
            [x_source_per_message, x_target_per_message], dim=1
        )

        return torch.nn.functional.elu(
            torch.matmul(x_source_target_per_message, self.att_weight)
        )

    def aggregate(self, x_message):
        """Aggregate messages on each target cell.

        A target cell receives messages from several source cells.
        This function aggregates these messages into a single output
        feature per target cell.

        🟧 This function corresponds to the within-neighborhood aggregation
        defined in [1]_ and [2]_.

        Parameters
        ----------
        x_message : torch.Tensor, shape = (..., n_messages, out_channels)
            Features associated with each message.
            One message is sent from a source cell to a target cell.

        Returns
        -------
        Tensor, shape = (...,  n_target_cells, out_channels)
            Output features on target cells.
            Each target cell aggregates messages from several source cells.
            Assumes that all target cells have the same rank s.
        """
        aggr = scatter(self.aggr_func)
        return aggr(x_message, self.target_index_i, 0)

    def forward(self, x_source, neighborhood, x_target=None):
        r"""Forward pass.

        This implements message passing for a given neighborhood:

        - from source cells with input features `x_source`,
        - via `neighborhood` defining where messages can pass,
        - to target cells with input features `x_target`.

        In practice, this will update the features on the target cells.

        If not provided, x_target is assumed to be x_source,
        i.e. source cells send messages to themselves.

        The message passing is decomposed into two steps:

        1. 🟥 Message: A message :math:`m_{y \rightarrow x}^{\left(r \rightarrow s\right)}`
        travels from a source cell :math:`y` of rank r to a target cell :math:`x` of rank s
        through a neighborhood of :math:`x`, denoted :math:`\mathcal{N} (x)`,
        via the message function :math:`M_\mathcal{N}`:

        .. math::
            m_{y \rightarrow x}^{\left(r \rightarrow s\right)}
                = M_{\mathcal{N}}\left(\mathbf{h}_x^{(s)}, \mathbf{h}_y^{(r)}, \Theta \right),

        where:

        - :math:`\mathbf{h}_y^{(r)}` are input features on the source cells, called `x_source`,
        - :math:`\mathbf{h}_x^{(s)}` are input features on the target cells, called `x_target`,
        - :math:`\Theta` are optional parameters (weights) of the message passing function.

        Optionally, attention can be applied to the message, such that:

        .. math::
            m_{y \rightarrow x}^{\left(r \rightarrow s\right)}
                \leftarrow att(\mathbf{h}_y^{(r)}, \mathbf{h}_x^{(s)}) . m_{y \rightarrow x}^{\left(r \rightarrow s\right)}

        2. 🟧 Aggregation: Messages are aggregated across source cells :math:`y` belonging to the
        neighborhood :math:`\mathcal{N}(x)`:

        .. math::
            m_x^{\left(r \rightarrow s\right)}
                = \text{AGG}_{y \in \mathcal{N}(x)} m_{y \rightarrow x}^{\left(r\rightarrow s\right)},

        resulting in the within-neighborhood aggregated message :math:`m_x^{\left(r \rightarrow s\right)}`.

        Details can be found in [1]_ and [2]_.

        Parameters
        ----------
        x_source : Tensor, shape = (..., n_source_cells, in_channels)
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        neighborhood : torch.sparse, shape = (n_target_cells, n_source_cells)
            Neighborhood matrix.
        x_target : Tensor, shape = (..., n_target_cells, in_channels)
            Input features on target cells.
            Assumes that all target cells have the same rank s.
            Optional. If not provided, x_target is assumed to be x_source,
            i.e. source cells send messages to themselves.

        Returns
        -------
        torch.Tensor, shape = (..., n_target_cells, out_channels)
            Output features on target cells.
            Assumes that all target cells have the same rank s.
        """
        neighborhood = neighborhood.coalesce()
        self.target_index_i, self.source_index_j = neighborhood.indices()
        neighborhood_values = neighborhood.values()

        x_message = self.message(x_source=x_source, x_target=x_target)
        x_message = x_message.index_select(-2, self.source_index_j)

        if self.att:
            attention_values = self.attention(x_source=x_source, x_target=x_target)
            neighborhood_values = torch.multiply(neighborhood_values, attention_values)

        x_message = neighborhood_values.view(-1, 1) * x_message
        return self.aggregate(x_message)


class Conv(MessagePassing):
    """Message passing: steps 1, 2, and 3.

    originally in topomodelx/base/conv.py

    Builds the message passing route given by one neighborhood matrix.
    Includes an option for an x-specific update function.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimension of output features.
    aggr_norm : bool, default=False
        Whether to normalize the aggregated message by the neighborhood size.
    update_func : {"relu", "sigmoid"}, optional
        Update method to apply to message.
    att : bool, default=False
        Whether to use attention.
    initialization : {"xavier_uniform", "xavier_normal"}, default="xavier_uniform"
        Initialization method.
    initialization_gain : float, default=1.414
        Initialization gain.
    with_linear_transform : bool, default=True
        Whether to apply a learnable linear transform.
        NB: if `False` in_channels has to be equal to out_channels.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        aggr_norm: bool = False,
        update_func: Literal["relu", "sigmoid", None] = None,
        att: bool = False,
        initialization: Literal["xavier_uniform", "xavier_normal"] = "xavier_uniform",
        initialization_gain: float = 1.414,
        with_linear_transform: bool = True,
    ) -> None:
        super().__init__(
            att=att,
            initialization=initialization,
            initialization_gain=initialization_gain,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr_norm = aggr_norm
        self.update_func = update_func

        self.weight = (
            Parameter(torch.Tensor(self.in_channels, self.out_channels))
            if with_linear_transform
            else None
        )

        if not with_linear_transform and in_channels != out_channels:
            raise ValueError(
                "With `linear_trainsform=False`, in_channels has to be equal to out_channels"
            )
        if self.att:
            self.att_weight = Parameter(
                torch.Tensor(
                    2 * self.in_channels,
                )
            )

        self.reset_parameters()

    def update(self, x_message_on_target) -> torch.Tensor:
        """Update embeddings on each cell (step 4).

        Parameters
        ----------
        x_message_on_target : torch.Tensor, shape = (n_target_cells, out_channels)
            Output features on target cells.

        Returns
        -------
        torch.Tensor, shape = (n_target_cells, out_channels)
            Updated output features on target cells.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(x_message_on_target)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x_message_on_target)
        return x_message_on_target

    def forward(self, x_source, neighborhood, x_target=None) -> torch.Tensor:
        """Forward pass.

        This implements message passing:
        - from source cells with input features `x_source`,
        - via `neighborhood` defining where messages can pass,
        - to target cells with input features `x_target`.

        In practice, this will update the features on the target cells.

        If not provided, x_target is assumed to be x_source,
        i.e. source cells send messages to themselves.

        Parameters
        ----------
        x_source : Tensor, shape = (..., n_source_cells, in_channels)
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        neighborhood : torch.sparse, shape = (n_target_cells, n_source_cells)
            Neighborhood matrix.
        x_target : Tensor, shape = (..., n_target_cells, in_channels)
            Input features on target cells.
            Assumes that all target cells have the same rank s.
            Optional. If not provided, x_target is assumed to be x_source,
            i.e. source cells send messages to themselves.

        Returns
        -------
        torch.Tensor, shape = (..., n_target_cells, out_channels)
            Output features on target cells.
            Assumes that all target cells have the same rank s.
        """
        if self.att:
            neighborhood = neighborhood.coalesce()
            self.target_index_i, self.source_index_j = neighborhood.indices()
            attention_values = self.attention(x_source, x_target)
            neighborhood = torch.sparse_coo_tensor(
                indices=neighborhood.indices(),
                values=attention_values * neighborhood.values(),
                size=neighborhood.shape,
            )
        if self.weight is not None:
            x_message = torch.mm(x_source, self.weight)
        else:
            x_message = x_source
        x_message_on_target = torch.mm(neighborhood, x_message)

        if self.aggr_norm:
            neighborhood_size = torch.sum(neighborhood.to_dense(), dim=1)
            x_message_on_target = torch.einsum(
                "i,ij->ij", 1 / neighborhood_size, x_message_on_target
            )

        return self.update(x_message_on_target)


class UniGINLayer(torch.nn.Module):
    """Layer of UniGIN.

    Originally in topomodelx/hypergraph/unigcn_layer.py

    Implementation of UniGIN layer proposed in [1]_.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    eps : float, default=0.0
        Constant in GIN Update equation.
    train_eps : bool, default=False
        Whether to make eps a trainable parameter.
    use_norm : bool, default=False
        Whether to apply row normalization after the layer.


    References
    ----------
    .. [1] Huang and Yang.
        UniGNN: a unified framework for graph and hypergraph neural networks.
        IJCAI 2021.
        https://arxiv.org/pdf/2105.00956.pdf
    .. [2] Papillon, Sanborn, Hajij, Miolane.
        Equations of topological neural networks (2023).
        https://github.com/awesome-tnns/awesome-tnns/
    .. [3] Papillon, Sanborn, Hajij, Miolane.
        Architectures of topological deep learning: a survey on topological neural networks (2023).
        https://arxiv.org/abs/2304.10031
    """

    def __init__(
        self,
        in_channels,
        eps: float = 0.0,
        train_eps: bool = False,
        use_norm: bool = False,
    ) -> None:
        super().__init__()

        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

        self.linear = torch.nn.Linear(in_channels, in_channels)

        self.use_norm = use_norm

        self.vertex2edge = Conv(
            in_channels=in_channels,
            out_channels=in_channels,
            with_linear_transform=False,
        )
        self.edge2vertex = Conv(
            in_channels=in_channels,
            out_channels=in_channels,
            with_linear_transform=False,
        )

    def forward(self, x_0, incidence_1):
        r"""[1]_ initially proposed the forward pass.

        Its equations are given in [2]_ and graphically illustrated in [3]_.

        The forward pass of this layer is composed of three steps.

        1. Every hyper-edge sums up the features of its constituent edges:

        ..  math::
            \begin{align*}
            &🟥 \quad m_{y \rightarrow z}^{(0 \rightarrow 1)}  = B_1^T \cdot h_y^{t, (0)}\\
            &🟧 \quad m_z^{(0 \rightarrow 1)}  = \sum_{y \in \mathcal{B}(z)} m_{y \rightarrow z}^{(0 \rightarrow 1)}\\
            \end{align*}

        2. The message to the nodes is the sum of the messages from the incident hyper-edges.

        .. math::
            \begin{align*}
            &🟥 \quad m_{z \rightarrow x}^{(1 \rightarrow 0)}  = B_1 \cdot m_z^{(0 \rightarrow 1)}\\
            &🟧 \quad m_{x}^{(1\rightarrow0)}  = \sum_{z \in \mathcal{C}(x)} m_{z \rightarrow x}^{(1\rightarrow0)}\\
            \end{align*}

        3. The node features are then updated using the GIN update equation:

        .. math::
            \begin{align*}
            &🟩 \quad m_x^{(0)}  = m_{x}^{(1\rightarrow0)}\\
            &🟦 \quad h_x^{t+1,(0)}  = \Theta^t \cdot ((1+\eps)\cdot h_x^{t,(0)}+m_x^{(0)})
            \end{align*}

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, in_channels)
            Input features on the nodes of the hypergraph.
        incidence_1 : torch.sparse, shape = (n_nodes, n_edges)
            Incidence matrix mapping edges to nodes (B_1).

        Returns
        -------
        x_0 : torch.Tensor
            Output node features.
        x_1 : torch.Tensor
            Output hyperedge features.
        """
        incidence_1_transpose = incidence_1.to_dense().T.to_sparse()
        # First pass fills in features of edges by adding features of constituent nodes
        x_1 = self.vertex2edge(x_0, incidence_1_transpose)
        # Second pass fills in features of nodes by adding features of the incident edges
        m_1_0 = self.edge2vertex(x_1, incidence_1)
        # Update node features using GIN update equation
        x_0 = self.linear((1 + self.eps) * x_0 + m_1_0)

        if self.use_norm:
            rownorm = x_0.detach().norm(dim=1, keepdim=True)
            scale = rownorm.pow(-1)
            scale[torch.isinf(scale)] = 0.0
            x_0 = x_0 * scale

        return x_0, x_1


class UniGCN(torch.nn.Module):
    """Neural network implementation of UniGCN [1]_ for hypergraph classification.

    Originally in topomodelx/hypergraph/unigcn.py
    UniGCN class.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    n_layers : int, default = 2
        Amount of message passing layers.

    References
    ----------
    .. [1] Huang and Yang.
        UniGNN: a unified framework for graph and hypergraph neural networks.
        IJCAI 2021.
        https://arxiv.org/pdf/2105.00956.pdf
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        n_layers=2,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            UniGCNLayer(
                in_channels=in_channels if i == 0 else hidden_channels,
                hidden_channels=hidden_channels,
            )
            for i in range(n_layers)
        )

    def forward(self, x_0, incidence_1):
        """Forward computation through layers, then linear layer, then global max pooling.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_edges, channels_edge)
            Edge features.

        incidence_1 : torch.Tensor, shape = (n_nodes, n_edges)
            Boundary matrix of rank 1.

        Returns
        -------
        x_0 : torch.Tensor
            Output node features.
        x_1 : torch.Tensor
            Output hyperedge features.
        """
        for layer in self.layers:
            x_0, x_1 = layer(x_0, incidence_1)

        return x_0, x_1


def from_sparse(data: _csc.csc_matrix) -> torch.Tensor:
    """Convert sparse input data directly to torch sparse coo format.

    Utils for more efficient sparse matrix casting to torch. Originally in topomodelx/utils/sparse.

    Parameters
    ----------
    data : scipy.sparse._csc.csc_matrix
        Input n_dimensional data.

    Returns
    -------
    torch.sparse_coo, same shape as data
        input data converted to tensor.
    """
    # cast from csc_matrix to coo format for compatibility
    coo = data.tocoo()

    values = torch.FloatTensor(coo.data)
    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))

    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()
