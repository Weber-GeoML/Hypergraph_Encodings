"""UniGNN-Compatible HGNN architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import configuration
from hgnn.hgnn_config import HGNNConfig

# Import only the modules that don't depend on torch_sparse


class HGNN(nn.Module):
    def __init__(
        self, H: torch.Tensor, in_size: int, out_size: int, config: HGNNConfig
    ):
        """Hypergraph Neural Network model compatible with M3."""
        super().__init__()

        self.W1 = nn.Linear(in_size, config.hidden_dims)
        self.W2 = nn.Linear(config.hidden_dims, out_size)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Convert to dense for M3 compatibility
        H_dense = H.to_dense() if H.is_sparse else H

        # Handle edge cases
        if H_dense.numel() == 0 or H_dense.sum() == 0:
            num_nodes = in_size
            identity = torch.eye(num_nodes, device=H.device)
            H_dense = torch.cat([identity, identity], dim=1)

        # Compute node degree
        d_V = H_dense.sum(1)
        d_V = torch.where(d_V == 0, torch.ones_like(d_V), d_V)

        # Compute edge degree
        d_E = H_dense.sum(0)
        d_E = torch.where(d_E == 0, torch.ones_like(d_E), d_E)

        # Compute Laplacian matrices
        D_v_invsqrt = torch.diag(d_V**-0.5)
        D_e_inv = torch.diag(d_E**-1)
        n_edges = d_E.shape[0]
        B = torch.eye(n_edges, device=H.device)

        # Compute Laplacian: L = D_v^{-1/2} H B D_e^{-1} H^T D_v^{-1/2}
        self.L = D_v_invsqrt @ H_dense @ B @ D_e_inv @ H_dense.T @ D_v_invsqrt

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass through the HGNN."""
        X = self.L @ self.W1(self.dropout(X))
        X = F.relu(X)
        X = self.L @ self.W2(self.dropout(X))
        return F.log_softmax(X, dim=1)
