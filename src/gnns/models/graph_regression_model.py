import torch
import torch.nn as nn
from measure_smoothing import dirichlet_normalized
from torch.nn import ModuleList, Dropout, ReLU, BatchNorm1d, Embedding, Linear, ModuleList, Sequential
from torch_geometric.nn import GCNConv, RGCNConv, SAGEConv, GatedGraphConv, GINConv, FiLMConv, global_mean_pool, GATConv, GINEConv, global_add_pool, GPSConv
import torch.nn.functional as F

from typing import Any, Dict, Optional
from models.performer import PerformerAttention


class RGATConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GATConv(in_features, out_features))
        self.convs = ModuleList(convs)
    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type==i]
            x_new += conv(x, rel_edge_index)
        return x_new

class RGINConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGINConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features))))
        self.convs = ModuleList(convs)
    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type==i]
            x_new += conv(x, rel_edge_index)
        return x_new

class GNN(torch.nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()
        self.args = args
        self.num_relations = args.num_relations
        self.layer_type = args.layer_type
        self.hidden_dim = args.hidden_dim
        num_features = [args.input_dim] + list(args.hidden_layers)# + [args.output_dim]
        self.num_layers = len(num_features) - 1
        layers = []
        for i, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            layers.append(self.get_layer(in_features, out_features))
        self.layers = ModuleList(layers)
        self.dropout = Dropout(p=args.dropout)
        self.act_fn = ReLU()

        if self.args.last_layer_fa:
            if self.layer_type == "R-GCN" or self.layer_type == "GCN":
                self.last_layer_transform = torch.nn.Linear(self.args.hidden_dim, self.args.output_dim)
            elif self.layer_type == "R-GIN" or self.layer_type == "GIN":
                self.last_layer_transform = nn.Sequential(nn.Linear(self.args.hidden_dim, self.args.hidden_dim),nn.BatchNorm1d(self.args.hidden_dim), nn.ReLU(),nn.Linear(self.args.hidden_dim, self.args.output_dim))
            else:
                raise NotImplementedError
            
        self.mlp = Sequential(
            Linear(self.hidden_dim, self.hidden_dim // 2),
            ReLU(),
            Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            ReLU(),
            Linear(self.hidden_dim // 4, self.args.output_dim),
        )

    def get_layer(self, in_features, out_features):
        if self.layer_type == "GCN":
            return GCNConv(in_features, out_features)
        elif self.layer_type == "R-GCN":
            return RGCNConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "R-GAT":
            return RGATConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "R-GIN":
            return RGINConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "GIN":
            return GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features)))
        elif self.layer_type == "SAGE":
            return SAGEConv(in_features, out_features)
        elif self.layer_type == "FiLM":
            return FiLMConv(in_features, out_features)

    def forward(self, x, edge_index, edge_attr, batch, measure_dirichlet=False):
        x = x.float()
        for i, layer in enumerate(self.layers):
            if self.layer_type in ["R-GCN", "R-GAT", "R-GIN", "FiLM"]:
                x_new = layer(x, edge_index, edge_type=graph.edge_type)
            else:
                x_new = layer(x, edge_index)
            if i != self.num_layers - 1:
                x_new = self.act_fn(x_new)
                x_new = self.dropout(x_new)
            if i == self.num_layers - 1 and self.args.last_layer_fa:
                combined_values = global_mean_pool(x, batch)
                combined_values = self.last_layer_transform(combined_values)
                if self.layer_type in ["R-GCN", "R-GIN"]:
                    x_new += combined_values[batch]
                else:
                    x_new = combined_values[batch]
            x = x_new 
        if measure_dirichlet:
            energy = dirichlet_normalized(x.cpu().numpy(), graph.edge_index.cpu().numpy())
            return energy
        x = global_mean_pool(x, batch)
        return self.mlp(x)
        # return x


class GINE(torch.nn.Module):
    """
    Create a GCN model for node classification
    with hidden layers of size 32.
    """
    # def __init__(self, channels=64, num_layers=4):
    def __init__(self, args):
        super().__init__()
        self.args = args
        hidden_dim = args.hidden_dim
        num_layers = args.num_layers
        output_dim = args.output_dim
        if output_dim == 1:
            # zinc
            input_dim = 28
            edge_dim = 4
        else:
            # peptides-struct
            input_dim = 9
            edge_dim = 3
        
        self.node_emb = Embedding(input_dim, hidden_dim)
        self.edge_emb = Embedding(edge_dim, hidden_dim)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
            )
            conv = GINConv(nn)
            # conv = GINEConv(nn)
            self.convs.append(conv)       
            
        self.mlp = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Linear(hidden_dim // 2, hidden_dim // 4),
            ReLU(),
            Linear(hidden_dim // 4, output_dim),
        )
        

    def forward(self, x, edge_index, edge_attr, batch):
        # x_pe = self.pe_norm(pe)
        # x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        # check if there is a gpu available
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
            x.to(device)
            edge_index.to(device)
            edge_attr.to(device)
            batch.to(device)
        """
        x = self.node_emb(x.squeeze(-1))
        attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            # x = conv(x, edge_index, batch, attr)
            x = conv(x, edge_index)
        x = global_add_pool(x, batch)
        return self.mlp(x)


class GPS(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        pe_dim = 20
        channels = args.hidden_dim
        input_dim = args.input_dim
        num_layers = len(list(args.hidden_layers)) + 1
        attn_type = 'performer'
        output_dim = args.output_dim
        if output_dim == 1:
            edge_dim = 4
        else:
            edge_dim = 3
        print("input_dim", input_dim)
        print("channels", channels)
        print("output_dim", output_dim)


        # self.node_emb = Linear(1, channels - pe_dim)
        self.node_emb = Embedding(input_dim, channels)
        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = BatchNorm1d(20)
        self.edge_emb = Embedding(edge_dim, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINConv(nn), heads=2)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, output_dim),
        )

        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    # def forward(self, x, pe, edge_index, edge_attr, batch):
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x.squeeze(-1))
        attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_add_pool(x, batch)
        return self.mlp(x)


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1