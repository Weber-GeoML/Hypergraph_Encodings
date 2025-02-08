import math

import pytest
import torch

from unignn_architectures.UniGCN import glorot, normalize_l2


@pytest.fixture
def sample_tensor():
    return torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)


def test_glorot_initialization(sample_tensor):
    glorot(sample_tensor)
    # Check if values are within expected range
    stdv = math.sqrt(6.0 / (sample_tensor.size(-2) + sample_tensor.size(-1)))
    assert torch.all(sample_tensor <= stdv)
    assert torch.all(sample_tensor >= -stdv)


def test_normalize_l2(sample_tensor):
    normalized = normalize_l2(sample_tensor)
    # Check if rows have unit norm
    row_norms = torch.norm(normalized, dim=1)
    assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-6)
