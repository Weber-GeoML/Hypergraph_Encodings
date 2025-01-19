import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from brec.dataset import BRECDataset
from encodings_hnns.encodings import HypergraphEncodings
from encodings_hnns.check_encodings_same import checks_encodings, test_laplacian, find_isomorphism_mapping
from encodings_hnns.liftings_and_expansions import lift_to_hypergraph
import os
import numpy as np
from itertools import permutations
from encodings_hnns.check_encodings_same import find_encoding_match

def create_output_dirs():
    """Create output directories for plots and results"""
    dirs = ['plots/graph_pairs', 'plots/hypergraph_pairs', 'plots/encodings', 'results']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)

def plot_graph_pair(graph1, graph2, pair_idx, category, is_isomorphic, output_dir):
    """Plot a pair of graphs side by side with clear isomorphism status"""
    plt.figure(figsize=(16, 8))
    
    # Create a colored box around the entire figure based on isomorphism status
    if is_isomorphic:
        plt.gca().patch.set_facecolor('lightgreen')
        plt.gca().patch.set_alpha(0.1)
        iso_status = "ISOMORPHIC"
        status_color = 'green'
    else:
        plt.gca().patch.set_facecolor('lightcoral')
        plt.gca().patch.set_alpha(0.1)
        iso_status = "NON-ISOMORPHIC"
        status_color = 'red'
    
    # Plot first graph
    plt.subplot(1, 2, 1)
    pos1 = nx.spring_layout(graph1)
    plt.title(f"Graph A\n{len(graph1.nodes())} nodes, {len(graph1.edges())} edges")
    nx.draw(graph1, pos1, 
           node_color='lightblue',
           node_size=500,
           with_labels=True,
           font_size=10,
           font_weight='bold')
    
    # Plot second graph
    plt.subplot(1, 2, 2)
    pos2 = nx.spring_layout(graph2)
    plt.title(f"Graph B\n{len(graph2.nodes())} nodes, {len(graph2.edges())} edges")
    nx.draw(graph2, pos2, 
           node_color='lightpink',
           node_size=500,
           with_labels=True,
           font_size=10,
           font_weight='bold')
    
    # Add main title with category and isomorphism status
    plt.suptitle(f"BREC Dataset - {category} Category\nPair {pair_idx}: {iso_status}", 
                fontsize=16, y=1.05,
                color=status_color,
                bbox=dict(facecolor='white', edgecolor=status_color, pad=10))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pair_{pair_idx}_{category.lower()}.png", 
                bbox_inches='tight',
                facecolor=plt.gca().get_facecolor(),
                edgecolor='none')
    plt.close()

def analyze_graph_pair(data1, data2, pair_idx, category, is_isomorphic):
    """Analyze a pair of graphs: plot them and compare their encodings"""
    # Convert PyG data to NetworkX graphs
    G1 = to_networkx(data1, to_undirected=True)
    G2 = to_networkx(data2, to_undirected=True)
    
    # Store the node mapping if graphs are isomorphic
    node_mapping = None
    if is_isomorphic:
        node_mapping = find_isomorphism_mapping(G1, G2)
        if node_mapping is None:
            print(f"WARNING: Pair {pair_idx} is marked as isomorphic but no isomorphism found!")
        else:
            print(f"\nIsomorphism mapping for pair {pair_idx}:")
            print("G1 node -> G2 node")
            for node1, node2 in node_mapping.items():
                print(f"{node1} -> {node2}")
    
    # Plot original graphs
    plot_graph_pair(G1, G2, pair_idx, category, is_isomorphic, 'plots/graph_pairs')
    
    # Convert to hypergraph dictionaries
    hg1 = convert_nx_to_hypergraph_dict(G1)
    hg2 = convert_nx_to_hypergraph_dict(G2)
    
    # Compare graph-level encodings
    print(f"\nAnalyzing pair {pair_idx} ({category}):")
    compare_encodings(hg1, hg2, pair_idx, category, is_isomorphic, "graph", node_mapping)
    
    # Lift to hypergraphs
    hg1_lifted = lift_to_hypergraph(data1)
    hg2_lifted = lift_to_hypergraph(data2)
    
    # Compare hypergraph-level encodings
    compare_encodings(hg1_lifted, hg2_lifted, pair_idx, category, is_isomorphic, "hypergraph", node_mapping)

def convert_nx_to_hypergraph_dict(G):
    """Convert NetworkX graph to hypergraph dictionary format"""
    hyperedges = {f"e_{i}": list(edge) for i, edge in enumerate(G.edges())}
    n = G.number_of_nodes()
    features = torch.empty((n, 0))
    return {"hypergraph": hyperedges, "features": features, "labels": {}, "n": n}

def compare_encodings(hg1, hg2, pair_idx, category, is_isomorphic, level="graph", node_mapping=None):
    """Compare encodings between two (hyper)graphs"""
    encoder1 = HypergraphEncodings()
    encoder2 = HypergraphEncodings()
    
    # Define encodings to check
    encodings_to_check = [
        ("LDP", "Local Degree Profile", True),
        ("LCP-FRC", "Local Curvature Profile - FRC", True),
        ("RWPE", "Random Walk Encodings", True),
        ("LCP-ORC", "Local Curvature Profile - ORC", False),
    ]
    
    output_dir = f'results/{level}_level'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/pair_{pair_idx}_{category.lower()}.txt', 'w') as f:
        f.write(f"Analysis for pair {pair_idx} ({category}) - {level} level\n")
        f.write(f"Isomorphic: {is_isomorphic}\n\n")
        
        for encoding_type, description, should_be_same in encodings_to_check:
            f.write(f"\n=== {description} ===\n")
            result = checks_encodings(
                encoding_type, 
                should_be_same, 
                hg1, 
                hg2, 
                encoder1, 
                encoder2, 
                "Graph A", 
                "Graph B",
                save_plots=True,
                plot_dir=f'plots/encodings/{level}/{pair_idx}',
                pair_idx=pair_idx,
                category=category,
                is_isomorphic=is_isomorphic,
                node_mapping=node_mapping
            )
            f.write(f"Result: {'Same' if result else 'Different'}\n")
        
        # Test Laplacian encodings
        f.write("\n=== Laplacian Encodings ===\n")
        for lap_type in ["Normalized", "RW", "Hodge"]:
            f.write(f"\nLaplacian type: {lap_type}\n")
            _, _, _, _, same = test_laplacian(
                hg1.copy(), 
                hg2.copy(), 
                lap_type,
                save_plots=True,
                plot_dir=f'plots/encodings/{level}/{pair_idx}'
            )
            f.write(f"Result: {'Same' if same else 'Different'}\n")


def plot_matched_encodings(encoding1, encoding2, ax1, ax2, title=""):
    """
    Plot two encodings, attempting to match their row orderings if possible.
    
    Args:
        encoding1, encoding2: numpy arrays of shape (n, d)
        ax1, ax2: matplotlib axes for plotting
        title: title for the plots
    """
    is_match, permuted, perm = find_encoding_match(encoding1, encoding2)
    
    if is_match:
        im1 = ax1.imshow(permuted)
        im2 = ax2.imshow(encoding2)
        ax1.set_title(f"{title}\nGraph A (Permuted)")
    else:
        im1 = ax1.imshow(encoding1)
        im2 = ax2.imshow(encoding2)
        ax1.set_title(f"{title}\nGraph A (Original)")
    
    ax2.set_title("Graph B")
    
    # Add colorbars
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    
    return is_match

def main():
    create_output_dirs()
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
        print(f"\nProcessing {category} category...")
        for pair_idx in range(start, end):
            # Get the pair of graphs
            graph1 = dataset[pair_idx * 2]
            graph2 = dataset[pair_idx * 2 + 1]
            
            # All pairs in BREC are non-isomorphic
            is_isomorphic = False
            
            # Analyze the pair
            analyze_graph_pair(graph1, graph2, pair_idx, category, is_isomorphic)

if __name__ == "__main__":
    main()