"""
This script is used to analyse the BREC dataset.
It is used to compare the encodings of the graphs in the BREC dataset.
"""

import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from brec.dataset import BRECDataset
from encodings_hnns.encodings import HypergraphEncodings
from encodings_hnns.check_encodings_same import checks_encodings, find_isomorphism_mapping
from encodings_hnns.liftings_and_expansions import lift_to_hypergraph
import os
import numpy as np
from itertools import permutations
from encodings_hnns.check_encodings_same import find_encoding_match
import hypernetx as hnx
from torch_geometric.data import Data

def create_output_dirs():
    """Create output directories for plots and results"""
    dirs = ['plots/graph_pairs', 'plots/hypergraph_pairs', 'plots/encodings', 'results']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)

def create_comparison_table(stats1, stats2):
    """Create comparison table with differences highlighted in red."""
    table_text = []
    colors = []  # List to store colors for each row
    
    for stat in stats1.keys():
        val1 = stats1[stat]
        val2 = stats2[stat]
        
        # Check if values are different
        is_different = False
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            is_different = not np.isclose(val1, val2, rtol=1e-5)
            row = f"{stat}:  {val1:.3f}  vs  {val2:.3f}"
        elif isinstance(val1, bool) and isinstance(val2, bool):
            is_different = val1 != val2
            row = f"{stat}:  {val1}  vs  {val2}"
        else:
            is_different = val1 != val2
            row = f"{stat}:  {val1}  vs  {val2}"
        
        table_text.append(row)
        colors.append('red' if is_different else 'black')
    
    return table_text, colors

def plot_hypergraph_pair(G1, G2, hg1, hg2, pair_idx, category, is_isomorphic, output_dir):
    """Plot comparison of two hypergraphs with their bipartite representations."""
    # Create figure with 4x2 subplot grid (increased height for new row)
    plt.figure(figsize=(30, 32))
    plt.suptitle(f"Pair {pair_idx} ({category})", fontsize=16)
    
    # Row 1: Original graphs
    # Plot first graph
    plt.subplot(421)
    pos1 = nx.circular_layout(G1)
    plt.title(f"Graph A\n{len(G1.nodes())} nodes, {len(G1.edges())} edges")
    nx.draw(G1, pos1, 
           node_color='lightblue',
           node_size=500,
           with_labels=True,
           font_size=10,
           font_weight='bold')
    
    # Plot second graph
    plt.subplot(422)
    pos2 = nx.circular_layout(G2)
    plt.title(f"Graph B\n{len(G2.nodes())} nodes, {len(G2.edges())} edges")
    nx.draw(G2, pos2, 
           node_color='lightpink',
           node_size=500,
           with_labels=True,
           font_size=10,
           font_weight='bold')
    ####################
    
    # Row 2: Hypergraph visualizations
    # Plot first hypergraph
    plt.subplot(423)
    H1 = hnx.Hypergraph(hg1['hypergraph'])
    hnx.draw(H1, 
            pos=pos1,
            with_node_labels=True,
            with_edge_labels=False,
            convex=False)
    plt.title(f"Hypergraph A\n({len(hg1['hypergraph'])} hyperedges)")
    
    # Plot second hypergraph
    plt.subplot(424)
    H2 = hnx.Hypergraph(hg2['hypergraph'])
    hnx.draw(H2, 
            pos=pos2,
            with_node_labels=True,
            with_edge_labels=False,
            convex=False)
    plt.title(f"Hypergraph B\n({len(hg2['hypergraph'])} hyperedges)")
    
    # Row 3: Bipartite representations
    # Plot first bipartite
    plt.subplot(425)
    BH1 = H1.bipartite()
    top1 = set(n for n, d in BH1.nodes(data=True) if d['bipartite'] == 0)
    pos1 = nx.bipartite_layout(BH1, top1)
    nx.draw(BH1, pos1, 
           with_labels=True,
           node_color=['lightblue' if node in top1 else 'lightgreen' for node in BH1.nodes()],
           node_size=500,
           font_size=12,
           font_weight='bold')
    plt.title(f"Graph A Bipartite")
    
    # Plot second bipartite
    plt.subplot(426)
    BH2 = H2.bipartite()
    top2 = set(n for n, d in BH2.nodes(data=True) if d['bipartite'] == 0)
    pos2 = nx.bipartite_layout(BH2, top2)
    nx.draw(BH2, pos2, 
           with_labels=True,
           node_color=['lightblue' if node in top2 else 'lightgreen' for node in BH2.nodes()],
           node_size=500,
           font_size=12,
           font_weight='bold')
    plt.title(f"Bipartite B")
    
    # Row 4: Hyperedge size distributions
    # Plot first histogram
    plt.subplot(427)
    sizes1 = [len(edge) for edge in hg1['hypergraph'].values()]
    plt.hist(sizes1, bins=range(min(sizes1), max(sizes1) + 2), 
            alpha=0.7, color='lightblue', rwidth=0.8)
    plt.title(f"Hypergraph A Size Distribution\n({len(hg1['hypergraph'])} total hyperedges)")
    plt.xlabel('Hyperedge Size')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    
    # Plot second histogram
    plt.subplot(428)
    sizes2 = [len(edge) for edge in hg2['hypergraph'].values()]
    plt.hist(sizes2, bins=range(min(sizes2), max(sizes2) + 2), 
            alpha=0.7, color='lightpink', rwidth=0.8)
    plt.title(f"Hypergraph B Size Distribution\n({len(hg2['hypergraph'])} total hyperedges)")
    plt.xlabel('Hyperedge Size')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    os.makedirs('plots/hypergraphs', exist_ok=True)
    plt.savefig(f'plots/hypergraphs/pair_{pair_idx}_{category.lower()}_hypergraphs.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Second figure: Statistics
    plt.figure(figsize=(10, 6))
    plt.title(f"Hypergraph Statistics - Pair {pair_idx} ({category})", pad=20)
    plt.axis('off')
    
    # Convert lists to tuples for hashing
    hyperedge_sizes1 = set(tuple(v) for v in hg1['hypergraph'].values())
    hyperedge_sizes2 = set(tuple(v) for v in hg2['hypergraph'].values())
    
    stats_text = [
        f"Graph A: {len(hg1['hypergraph'])} hyperedges",
        f"Graph B: {len(hg2['hypergraph'])} hyperedges",
        f"\nHyperedge sizes Graph A:",
        *[f"Size {len(v)}: {sum(1 for e in hg1['hypergraph'].values() if len(e) == len(v))}" 
          for v in hyperedge_sizes1],
        f"\nHyperedge sizes Graph B:",
        *[f"Size {len(v)}: {sum(1 for e in hg2['hypergraph'].values() if len(e) == len(v))}" 
          for v in hyperedge_sizes2]
    ]
    plt.text(0.1, 0.5, '\n'.join(stats_text), fontsize=12, transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(f'plots/hypergraphs/pair_{pair_idx}_{category.lower()}_statistics.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_graph_pair(graph1, graph2, pair_idx, category, is_isomorphic, output_dir):
    """Plot a pair of graphs side by side with graph statistics and degree distributions."""
    # Create figure with 3x2 subplot grid (added row for adjacency matrices)
    fig = plt.figure(figsize=(16, 28))
    
    # Set isomorphism status
    iso_status = "ISOMORPHIC" if is_isomorphic else "NON-ISOMORPHIC"
    status_color = 'green' if is_isomorphic else 'red'
    
    # Plot first graph
    ax1 = plt.subplot(3, 2, 1)
    pos1 = nx.spring_layout(graph1)
    plt.title(f"Graph A\n{len(graph1.nodes())} nodes, {len(graph1.edges())} edges")
    nx.draw(graph1, pos1, 
           node_color='lightblue',
           node_size=500,
           with_labels=True,
           font_size=10,
           font_weight='bold')
    
    # Plot second graph
    ax2 = plt.subplot(3, 2, 2)
    pos2 = nx.spring_layout(graph2)
    plt.title(f"Graph B\n{len(graph2.nodes())} nodes, {len(graph2.edges())} edges")
    nx.draw(graph2, pos2, 
           node_color='lightpink',
           node_size=500,
           with_labels=True,
           font_size=10,
           font_weight='bold')
    
    # Plot degree distributions
    ax3 = plt.subplot(3, 2, 3)
    degrees1 = [d for n, d in graph1.degree()]
    degrees2 = [d for n, d in graph2.degree()]
    max_degree = max(max(degrees1), max(degrees2))
    min_degree = min(min(degrees1), min(degrees2))
    bins = range(min_degree, max_degree + 2)  # +2 to include max degree
    
    # Set the width and positions for the bars
    width = 0.35  # Width of the bars
    x = np.array(list(bins[:-1]))  # Bar positions for graph1
    
    # Create the bars with offset positions
    plt.bar(x - width/2, np.histogram(degrees1, bins=bins)[0], 
            width, alpha=0.7, color='lightblue', label='Graph A')
    plt.bar(x + width/2, np.histogram(degrees2, bins=bins)[0], 
            width, alpha=0.7, color='lightpink', label='Graph B')
    
    plt.title(f"Degree Distribution\n({len(graph1.nodes())} total nodes)")
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text with exact counts for Graph A
    unique_degrees1 = sorted(set(degrees1))
    degree_counts1 = {deg: degrees1.count(deg) for deg in unique_degrees1}
    text1 = 'Graph A:\n' + '\n'.join([f'Degree {deg}: {count}' for deg, count in degree_counts1.items()])
    plt.text(0.95, 0.95, text1,
             transform=ax3.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8),
             fontsize=8)
    
    # Plot second histogram
    ax3.hist(degrees2, bins=bins, 
            alpha=0.7, color='lightpink', rwidth=0.8,
            label='Graph B')
    
    # Add text with exact counts for Graph B
    unique_degrees2 = sorted(set(degrees2))
    degree_counts2 = {deg: degrees2.count(deg) for deg in unique_degrees2}
    text2 = 'Graph B:\n' + '\n'.join([f'Degree {deg}: {count}' for deg, count in degree_counts2.items()])
    plt.text(0.75, 0.95, text2,
             transform=ax3.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8),
             fontsize=8)
    
    # Compute and display graph statistics
    ax4 = plt.subplot(3, 2, 4)
    ax4.axis('off')
    
    def get_graph_stats(G):
        stats = {
            # Basic statistics
            'Number of nodes': G.number_of_nodes(),
            'Number of edges': G.number_of_edges(),
            'Average degree': np.mean([d for n, d in G.degree()]),
            'Maximum degree': max([d for n, d in G.degree()]),
            'Minimum degree': min([d for n, d in G.degree()]),
            'Density': nx.density(G),
            
            # Structural properties
            'Is bipartite': nx.is_bipartite(G),
            'Number of triangles': sum(nx.triangles(G).values()) // 3,
            # Convert generator to list to count cliques
            # number of maximal cliques
            'Number of maximal cliques': sum(1 for c in nx.find_cliques(G)),
            # the largest maximal clique
            'Largest maximal clique': max(nx.find_cliques(G), key=len),
            'Edge connectivity': nx.edge_connectivity(G),
            'Node connectivity': nx.node_connectivity(G),
            'Number of components': nx.number_connected_components(G),
            'Is planar': nx.is_planar(G),
            
            # Centrality measures (averaged over nodes)
            'Avg betweenness': np.mean(list(nx.betweenness_centrality(G).values())),
            'Avg closeness': np.mean(list(nx.closeness_centrality(G).values())),
            'Avg eigenvector': np.mean(list(nx.eigenvector_centrality_numpy(G).values())),
            
            # Spectral properties
            'Spectral radius': max(abs(nx.adjacency_spectrum(G))),
            'Algebraic connectivity': nx.algebraic_connectivity(G),
            'Spectral gap': sorted(abs(nx.adjacency_spectrum(G)))[-1] - sorted(abs(nx.adjacency_spectrum(G)))[-2],
        }
        
        # Add these stats only if graph is connected
        if nx.is_connected(G):
            stats.update({
                'Diameter': nx.diameter(G),
                'Average shortest path': nx.average_shortest_path_length(G),
                'Average clustering': nx.average_clustering(G),
                'Assortativity': nx.degree_assortativity_coefficient(G),
                'Radius': nx.radius(G),
                'Center size': len(nx.center(G)),
                'Periphery size': len(nx.periphery(G))
            })
            try:
                stats['Girth'] = len(min(nx.cycle_basis(G), key=len)) if nx.cycle_basis(G) else float('inf')
            except:
                stats['Girth'] = 'N/A'
        else:
            stats.update({
                'Diameter': 'N/A (disconnected)',
                'Average shortest path': 'N/A (disconnected)',
                'Average clustering': nx.average_clustering(G),
                'Assortativity': nx.degree_assortativity_coefficient(G),
                'Girth': 'N/A (disconnected)',
                'Radius': 'N/A (disconnected)',
                'Center size': 'N/A (disconnected)',
                'Periphery size': 'N/A (disconnected)'
            })
        
        return stats
    
    stats1 = get_graph_stats(graph1)
    stats2 = get_graph_stats(graph2)
    
    # Create comparison table with colored differences
    table_text, colors = create_comparison_table(stats1, stats2)
    
    # Plot statistics with colored differences - smaller text and spacing
    y_pos = 0.98
    line_height = 0.025  # Reduced from 0.04
    
    # Plot title
    ax4.text(0.05, y_pos, 'Graph Statistics:', 
             fontsize=8,  # Reduced from 10
             family='monospace',
             verticalalignment='top', transform=ax4.transAxes,
             color='black', fontweight='bold')
    
    # Plot each statistic with appropriate color
    y_pos -= line_height  # Reduced space after title
    for text, color in zip(table_text, colors):
        ax4.text(0.05, y_pos, text,
                fontsize=7,  # Reduced from 10
                family='monospace',
                verticalalignment='top', transform=ax4.transAxes,
                color=color)
        y_pos -= line_height
    
    # Add main title
    plt.suptitle(f"BREC Dataset - {category} Category\nPair {pair_idx}: {iso_status}", 
                fontsize=16, y=1.02,
                color=status_color,
                bbox=dict(facecolor='white', edgecolor=status_color, pad=10))
    
    # Add adjacency matrix plots in new row
    ax5 = plt.subplot(3, 2, 5)
    adj1 = nx.adjacency_matrix(graph1).todense()
    adj2 = nx.adjacency_matrix(graph2).todense()
    
    # Plot first adjacency matrix
    im1 = ax5.imshow(adj1, cmap='viridis')
    plt.colorbar(im1, ax=ax5)
    ax5.set_title('Graph A Adjacency Matrix')
    ax5.set_xlabel('Node Index')
    ax5.set_ylabel('Node Index')
    
    # Plot second adjacency matrix
    ax6 = plt.subplot(3, 2, 6)
    im2 = ax6.imshow(adj2, cmap='viridis')
    plt.colorbar(im2, ax=ax6)
    ax6.set_title('Graph B Adjacency Matrix')
    ax6.set_xlabel('Node Index')
    ax6.set_ylabel('Node Index')
    
    # Create separate figure for difference plot
    plt.figure(figsize=(8, 6))
    # Compute difference (padding smaller matrix if sizes differ)
    max_size = max(adj1.shape[0], adj2.shape[0])
    padded1 = np.pad(adj1, ((0, max_size - adj1.shape[0]), (0, max_size - adj1.shape[0])))
    padded2 = np.pad(adj2, ((0, max_size - adj2.shape[0]), (0, max_size - adj2.shape[0])))
    diff = padded1 - padded2
    
    im3 = plt.imshow(diff, cmap='Blues', vmin=-1, vmax=1)
    plt.colorbar(im3)
    plt.title('Adjacency Matrix Difference (A - B)')
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    
    # Save difference plot
    plt.tight_layout()
    os.makedirs(f'{output_dir}/adjacency_diffs', exist_ok=True)
    plt.savefig(f"{output_dir}/adjacency_diffs/pair_{pair_idx}_{category.lower()}_diff.png",
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Return to main figure and finish
    plt.figure(fig.number)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pair_{pair_idx}_{category.lower()}.png",
                bbox_inches='tight', dpi=300)
    plt.close()

def analyze_graph_pair(data1, data2, pair_idx, category, is_isomorphic):
    """Analyze a pair of graphs: plot them and compare their encodings"""
    # Convert PyG data to NetworkX graphs
    G1 = to_networkx(data1, to_undirected=True)
    G2 = to_networkx(data2, to_undirected=True)
    assert not is_isomorphic, "All pairs in BREC are non-isomorphic"

    # store the Asjacency matrix plots and their difference
    
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
    # in graph space
    plot_graph_pair(G1, G2, pair_idx, category, is_isomorphic, 'plots/graph_pairs')
    
    # Convert to hypergraph dictionaries
    # THESE ARE STILL GRAPHS!!!
    hg1 = convert_nx_to_hypergraph_dict(G1)
    hg2 = convert_nx_to_hypergraph_dict(G2)

    
    # Compare graph-level encodings
    print(f"\nAnalyzing pair {pair_idx} ({category}):")
    print("\n")
    compare_encodings(hg1, hg2, pair_idx, category, is_isomorphic, "graph", node_mapping)

    del hg1, hg2

    print('*-'*25)
    print('*-'*25)
    print(f"Analyzing pair {pair_idx} ({category}): at the hypergraph level")
    print('*-'*25)
    print('*-'*25)
    
    # Lift to hypergraphs
    hg1_lifted = lift_to_hypergraph(data1, verbose=False)
    hg2_lifted = lift_to_hypergraph(data2, verbose=False)

    plot_hypergraph_pair(G1, G2, hg1_lifted, hg2_lifted, pair_idx, category, is_isomorphic, 'plots/hypergraph_pairs')
    
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
    assert not is_isomorphic, "All pairs in BREC are non-isomorphic"
    
    # Define encodings to check
    encodings_to_check = [
        ("LDP", "Local Degree Profile", True),
        ("LCP-FRC", "Local Curvature Profile - FRC", True),
        ("RWPE", "Random Walk Encodings", True),
        ("LCP-ORC", "Local Curvature Profile - ORC", False),
        ("LAPE-Normalized", "Normalized Laplacian", True),
        ("LAPE-RW", "Random Walk Laplacian", True),
        ("LAPE-Hodge", "Hodge Laplacian", True),
    ]
    
    output_dir = f'results/{level}_level'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    with open(f'{output_dir}/pair_{pair_idx}_{category.lower()}.txt', 'w') as f:
        f.write(f"Analysis for pair {pair_idx} ({category}) - {level} level\n")
        f.write(f"Isomorphic: {is_isomorphic}\n\n")
        
        for encoding_type, description, should_be_same in encodings_to_check:
            f.write(f"\n=== {description} ===\n")
            result = checks_encodings(
                name_of_encoding=encoding_type, 
                same=should_be_same, 
                hg1=hg1, 
                hg2=hg2, 
                encoder_shrikhande=encoder1, 
                encoder_rooke=encoder2, 
                name1="Graph A", 
                name2="Graph B",
                save_plots=True,
                plot_dir=f'plots/encodings/{level}/{pair_idx}',
                pair_idx=pair_idx,
                category=category,
                is_isomorphic=is_isomorphic,
                node_mapping=node_mapping,
                graph_type=level,
            )
            f.write(f"Result: {'Same' if result else 'Different'}\n")


def main():
    create_output_dirs()
    dataset = BRECDataset()

    # First analyze Rook and Shrikhande graphs
    print("\nAnalyzing Rook and Shrikhande graphs...")
    
    # Load the graphs
    rook = nx.read_graph6("rook_graph.g6")
    shrikhande = nx.read_graph6("shrikhande.g6")
    
    # Convert to PyG Data objects
    def nx_to_pyg(G):
        edge_index = torch.tensor([[e[0] for e in G.edges()], 
                                 [e[1] for e in G.edges()]], dtype=torch.long)
        x = torch.ones((G.number_of_nodes(), 1), dtype=torch.float)
        y = torch.zeros(G.number_of_nodes(), dtype=torch.long)
        return Data(x=x, y=y, edge_index=edge_index, num_nodes=G.number_of_nodes())
    
    rook_data = nx_to_pyg(rook)
    shrikhande_data = nx_to_pyg(shrikhande)
    
    # Analyze as a special pair
    print("Analyzing Rook vs Shrikhande")
    analyze_graph_pair(
        rook_data, 
        shrikhande_data, 
        pair_idx="rook_vs_shrikhande", 
        category="Special", 
        is_isomorphic=False
    )

    
    # Then continue with BREC dataset analysis
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
            print(f"Processing pair {pair_idx}...")
            # Get the pair of graphs
            graph1 = dataset[pair_idx * 2]
            graph2 = dataset[pair_idx * 2 + 1]
            
            # All pairs in BREC are non-isomorphic
            is_isomorphic = False
            
            # Analyze the pair
            analyze_graph_pair(graph1, graph2, pair_idx, category, is_isomorphic)

if __name__ == "__main__":
    main()