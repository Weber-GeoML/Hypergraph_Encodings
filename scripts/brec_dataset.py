"""
This script is used to plot the BREC dataset.
This is a small example of how to plot the graphs in the BREC dataset.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

from brec.dataset import BRECDataset
from brec.evaluator import evaluate


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 16)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


model = GCN()

# Function to plot a PyG graph
def plot_graph(data, index):
    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(8, 8))
    plt.title(f"Graph {index}")
    
    # Use spring layout for node positioning
    pos = nx.spring_layout(G)
    
    # Draw the graph
    nx.draw(G, pos, 
           node_color='lightblue',
           node_size=500,
           with_labels=True,
           font_size=10,
           font_weight='bold')
    
    plt.show()

def get_pair_category_and_isomorphism(pair_index):
    part_dict = {
        "Basic": (0, 60),
        "Regular": (60, 160),
        "Extension": (160, 260),
        "CFI": (260, 360),
        "4-Vertex_Condition": (360, 380),
        "Distance_Regular": (380, 400),
    }
    
    # Find which category this pair belongs to
    for category, (start, end) in part_dict.items():
        if start <= pair_index < end:
            is_isomorphic = False
            return category, is_isomorphic
    
    return "Unknown", False

def plot_graph_pair(data1, data2, pair_index):
    plt.figure(figsize=(16, 8))
    
    # Get category and isomorphism status
    category, is_isomorphic = get_pair_category_and_isomorphism(pair_index)
    iso_status = "Isomorphic" if is_isomorphic else "Non-isomorphic"
    
    # Plot first graph
    plt.subplot(1, 2, 1)
    G1 = to_networkx(data1, to_undirected=True)
    pos1 = nx.spring_layout(G1)
    plt.title(f"Graph Pair {pair_index} - A\n({iso_status})")
    nx.draw(G1, pos1, 
           node_color='lightblue',
           node_size=500,
           with_labels=True,
           font_size=10,
           font_weight='bold')
    
    # Plot second graph
    plt.subplot(1, 2, 2)
    G2 = to_networkx(data2, to_undirected=True)
    pos2 = nx.spring_layout(G2)
    plt.title(f"Graph Pair {pair_index} - B\n({iso_status})")
    nx.draw(G2, pos2, 
           node_color='lightpink',
           node_size=500,
           with_labels=True,
           font_size=10,
           font_weight='bold')
    
    # Add a global title with category and isomorphism information
    plt.suptitle(f"Category: {category}\nPair {pair_index}: {iso_status} Graphs", fontsize=16, y=1.05)
    
    plt.tight_layout()
    plt.show()

def plot_examples_from_each_category():
    dataset = BRECDataset()
    part_dict = {
        "Basic": (0, 60),
        "Regular": (60, 160),
        "Extension": (160, 260),
        "CFI": (260, 360),
        "4-Vertex_Condition": (360, 380),
        "Distance_Regular": (380, 400),
    }
    
    for category, (start, end) in part_dict.items():
        # Get two different non-isomorphic pairs from the category
        pair_idx_1 = start + (end - start) // 4  # Take from first quarter
        pair_idx_2 = start + (end - start) * 3 // 4  # Take from last quarter
        
        graph1_pair1 = dataset[pair_idx_1 * 2]
        graph2_pair1 = dataset[pair_idx_1 * 2 + 1]
        
        graph1_pair2 = dataset[pair_idx_2 * 2]
        graph2_pair2 = dataset[pair_idx_2 * 2 + 1]
        
        # Plot both pairs
        plt.figure(figsize=(20, 10))
        plt.suptitle(f"BREC Dataset - {category} Category\nNon-isomorphic Pairs {pair_idx_1} and {pair_idx_2}", 
                    fontsize=16, y=1.05)
        
        # Plot first non-isomorphic pair
        plt.subplot(2, 2, 1)
        G1_pair1 = to_networkx(graph1_pair1, to_undirected=True)
        pos1_pair1 = nx.spring_layout(G1_pair1)
        plt.title(f"{category} - Pair {pair_idx_1} - Graph A")
        nx.draw(G1_pair1, pos1_pair1, 
               node_color='lightblue',
               node_size=500,
               with_labels=True,
               font_size=10,
               font_weight='bold')
        
        plt.subplot(2, 2, 2)
        G2_pair1 = to_networkx(graph2_pair1, to_undirected=True)
        pos2_pair1 = nx.spring_layout(G2_pair1)
        plt.title(f"{category} - Pair {pair_idx_1} - Graph B")
        nx.draw(G2_pair1, pos2_pair1, 
               node_color='lightblue',
               node_size=500,
               with_labels=True,
               font_size=10,
               font_weight='bold')
        
        # Plot second non-isomorphic pair
        plt.subplot(2, 2, 3)
        G1_pair2 = to_networkx(graph1_pair2, to_undirected=True)
        pos1_pair2 = nx.spring_layout(G1_pair2)
        plt.title(f"{category} - Pair {pair_idx_2} - Graph A")
        nx.draw(G1_pair2, pos1_pair2, 
               node_color='lightpink',
               node_size=500,
               with_labels=True,
               font_size=10,
               font_weight='bold')
        
        plt.subplot(2, 2, 4)
        G2_pair2 = to_networkx(graph2_pair2, to_undirected=True)
        pos2_pair2 = nx.spring_layout(G2_pair2)
        plt.title(f"{category} - Pair {pair_idx_2} - Graph B")
        nx.draw(G2_pair2, pos2_pair2, 
               node_color='lightpink',
               node_size=500,
               with_labels=True,
               font_size=10,
               font_weight='bold')
        
        plt.tight_layout()
        plt.show()

# Call the function to plot examples
plot_examples_from_each_category()