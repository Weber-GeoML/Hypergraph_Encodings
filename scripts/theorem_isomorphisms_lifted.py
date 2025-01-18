from encodings_hnns.encodings import HypergraphEncodings


import networkx as nx
import itertools
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np


import networkx as nx
from encodings_hnns.encodings import HypergraphEncodings

def lift_and_compare_encodings():
    """Lift graphs to hypergraphs and compare their encodings"""
    
    # Load and convert graphs
    shrikhande = nx.read_graph6("shrikhande.g6")
    rooke = nx.read_graph6("rook_graph.g6")
    
    # Convert to PyG Data objects
    def nx_to_pyg(G):
        edge_index = torch.tensor([[e[0] for e in G.edges()], 
                                 [e[1] for e in G.edges()]], dtype=torch.long)
        x = torch.ones((G.number_of_nodes(), 1), dtype=torch.float)
        y = torch.zeros(G.number_of_nodes(), dtype=torch.long)
        return Data(x=x, y=y, edge_index=edge_index, num_nodes=G.number_of_nodes())
    
    # Convert and lift graphs to hypergraphs
    shrikhande_hyper = lift_to_hypergraph(nx_to_pyg(shrikhande))
    rooke_hyper = lift_to_hypergraph(nx_to_pyg(rooke))
    
    # Initialize encoder
    encoder_shrikhande = HypergraphEncodings()
    encoder_rooke = HypergraphEncodings()

    # Define encodings to check with their parameters
    encodings_to_check = [
        ("LDP", "Local Degree Profile", True),
        ("LCP-FRC", "Local Curvature Profile - FRC", True),
        ("RWPE", "Random Walk Encodings", True),
        ("LCP-ORC", "Local Curvature Profile - ORC", False),
    ]

    print("\nComparing encodings for lifted hypergraphs:")
    
    # Check each encoding type
    for encoding_type, description, should_be_same in encodings_to_check:
        print(f"\n=== {description} ===")
        checks_encodings(
            encoding_type, 
            should_be_same, 
            shrikhande_hyper, 
            rooke_hyper, 
            encoder_shrikhande, 
            encoder_rooke, 
            "Shrikhande (lifted)", 
            "Rooke (lifted)"
        )

    # Test Laplacian encodings
    print("\n=== Laplacian Encodings ===")
    for lap_type in ["Normalized", "RW", "Hodge"]:
        print(f"\nLaplacian type: {lap_type}")
        hg1_lape, hg2_lape, L1, L2, same = test_laplacian(
            shrikhande_hyper.copy(), 
            rooke_hyper.copy(), 
            lap_type, 
            name1="Shrikhande (lifted)", 
            name2="Rooke (lifted)"
        )

if __name__ == "__main__":
    # Original lifting and plotting
    lift_and_plot_graphs()
    
    # New encoding comparison
    lift_and_compare_encodings()