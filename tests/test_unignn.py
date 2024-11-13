import pytest
import torch
from uniGCN.UniGCN import UniGNN


@pytest.fixture
def args():
    class Args:
        def __init__(self):
            self.model_name = "UniGCN"
            self.use_norm = True
            self.first_aggregate = "mean"
            self.second_aggregate = "sum"
            self.activation = "relu"
            self.input_drop = 0.1
            self.dropout = 0.1
            self.attn_drop = 0.1

    return Args()


@pytest.fixture
def simple_data():
    # Create simple test data
    X = torch.randn(5, 3)  # 5 nodes, 3 features
    V = torch.tensor([0, 1, 1, 2, 2, 3, 4])
    E = torch.tensor([0, 0, 1, 1, 2, 2, 2])
    return X, V, E


def test_unignn_initialization(args):
    model = UniGNN(args, nfeat=3, nhid=4, nclass=2, nlayer=2, nhead=2)
    assert len(model.convs) == 1  # One conv layer
    assert isinstance(model.conv_out, torch.nn.Module)


# def test_unignn_forward(args, simple_data):
#     X, V, E = simple_data
#     model = UniGNN(args, nfeat=3, nhid=4, nclass=2, nlayer=2, nhead=2)

#     output = model(X, V, E)

#     # Check output shape and properties
#     assert output.shape == (5, 2)  # 5 nodes, 2 classes
#     assert torch.allclose(
#         torch.exp(output).sum(dim=1), torch.ones(5)
#     )  # Valid probability distribution
