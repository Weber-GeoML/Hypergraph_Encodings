"""
This script is used to analyse the BREC dataset.
It is used to compare the encodings of the graphs in the BREC dataset.
"""

import networkx as nx
from brec.dataset import BRECDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from encodings_hnns.liftings_and_expansions import lift_to_hypergraph
from brec_analysis.utils_for_brec import create_output_dirs, convert_nx_to_hypergraph_dict, nx_to_pyg
from brec_analysis.plotting_graphs_and_hgraphs_for_brec import plot_graph_pair, plot_hypergraph_pair
from brec_analysis.compare_encodings_wrapper import compare_encodings_wrapper
from brec_analysis.match_status import MatchStatus
import json
import matplotlib.pyplot as plt
import os
import click
from brec_analysis.encodings_to_check import ENCODINGS_TO_CHECK
import multiprocessing as mp
from functools import partial

# Define categories and their ranges
part_dict: dict[str, tuple[int, int]] = {
    # "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 360),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
}

def analyze_graph_pair(
    data1: Data, data2: Data, pair_idx: int|str, category: str, is_isomorphic: bool, encoding: str
) -> dict:
    """Analyze a pair of graphs: plot them and compare their encodings
    
    Args:
        data1 (Data):
            The first graph.
        data2 (Data):
            The second graph.
        pair_idx (str):
            The index of the pair.
        category (str):
            The category of the pair.
        is_isomorphic (bool):
            Whether the graphs are isomorphic.
        encoding (str):
            The encoding to use.
    """
    # Convert PyG data to NetworkX graphs
    G1 = to_networkx(data1, to_undirected=True)
    G2 = to_networkx(data2, to_undirected=True)
    assert not is_isomorphic, "All pairs in BREC are non-isomorphic"

    # store the Asjacency matrix plots and their difference

    # Store the node mapping if graphs are isomorphic
    node_mapping = None
    if is_isomorphic:
        node_mapping = find_isomorphism_mapping(G1, G2)
        if node_mapping is None:
            print(
                f"WARNING: Pair {pair_idx} is marked as isomorphic but no isomorphism found!"
            )
        else:
            print(f"\nIsomorphism mapping for pair {pair_idx}:")
            print("G1 node -> G2 node")
            for node1, node2 in node_mapping.items():
                print(f"{node1} -> {node2}")

    # Plot original graphs
    # in graph space
    plot_graph_pair(
        G1, G2, pair_idx, category, is_isomorphic, "plots/graph_pairs"
    )

    # Convert to hypergraph dictionaries
    # THESE ARE STILL GRAPHS!!!
    hg1 = convert_nx_to_hypergraph_dict(G1)
    hg2 = convert_nx_to_hypergraph_dict(G2)

    # Initialize the results structure
    final_results : dict = {
        "pair_idx": pair_idx,
        "category": category,
        "is_isomorphic": is_isomorphic,
        "graph_level": {
            "encodings": {}
        },
        "hypergraph_level": {
            "encodings": {}
        }
    }

    # Compare graph-level encodings
    print(f"\nAnalyzing pair {pair_idx} ({category}):")
    print("\n")
    graph_results = compare_encodings_wrapper(
        hg1, hg2, pair_idx, category, is_isomorphic, "graph", node_mapping
    )
    final_results["graph_level"]["encodings"] = graph_results["encodings"]

    del hg1, hg2

    print("*-" * 25)
    print("*-" * 25)
    print(f"Analyzing pair {pair_idx} ({category}): at the hypergraph level")
    print("*-" * 25)
    print("*-" * 25)

    # Lift to hypergraphs and compare
    hg1_lifted = lift_to_hypergraph(data1, verbose=False)
    hg2_lifted = lift_to_hypergraph(data2, verbose=False)

    plot_hypergraph_pair(
        G1,
        G2,
        hg1_lifted,
        hg2_lifted,
        pair_idx,
        category,
        is_isomorphic,
        "plots/hypergraph_pairs",
    )

    # Compare hypergraph-level encodings
    hypergraph_results = compare_encodings_wrapper(
        hg1_lifted,
        hg2_lifted,
        pair_idx,
        category,
        is_isomorphic,
        level="hypergraph",
        node_mapping=node_mapping,
        encoding_type=encoding,
    )
    final_results["hypergraph_level"]["encodings"] = hypergraph_results["encodings"]

    return final_results

def write_results(f, results: dict, json_results: dict) -> None:
    """Write results for a pair to the file and update JSON results."""
    # Print the file path at the start
    print(f"\nWriting results to: {f.name}")
    
    # Get pair info
    pair_idx = results['pair_idx']
    category = results['category']
    
    # Initialize this pair in json_results if needed
    if category not in json_results:
        json_results[category] = {}
    
    for level in ["graph_level", "hypergraph_level"]:
        f.write(f"\nAnalysis for pair {pair_idx} ({category}) - {level}\n")
        
        for encoding_type, result in results[level]["encodings"].items():
            # Write to text file
            f.write(f"\n=== {result[level]['encodings']} ===\n")
            is_same = result['status'] in [MatchStatus.EXACT_MATCH, MatchStatus.SCALED_MATCH]
            f.write(f"Result: {'Same' if is_same else 'Different'}\n")
            if result['status'] == MatchStatus.SCALED_MATCH:
                f.write(f"Scaling factor: {result['scaling_factor']}\n")
            
            # Update JSON structure
            encoding_key = f"{level.split('_')[0].capitalize()} ({encoding_type})"
            if encoding_key not in json_results[category]:
                json_results[category][encoding_key] = {
                    'different': 0,
                    'total': 0
                }
            
            json_results[category][encoding_key]['total'] += 1
            if not is_same:
                json_results[category][encoding_key]['different'] += 1

def plot_disconnected_pair(G1: nx.Graph, G2: nx.Graph, pair_idx: int, category: str) -> None:
    """Plot a pair of graphs side by side, highlighting their connectivity status.
    
    Args:
        G1: First graph
        G2: Second graph
        pair_idx: Index of the pair
        category: Category of the pair
    """
    plt.figure(figsize=(12, 5))
    
    # Plot first graph
    plt.subplot(121)
    pos1 = nx.spring_layout(G1, seed=42)
    
    # Color nodes based on connected components for G1
    components1 = list(nx.connected_components(G1))
    colors1 = [f'C{i}' for i in range(len(components1))]
    node_colors1 = ['white'] * G1.number_of_nodes()
    for comp_idx, component in enumerate(components1):
        for node in component:
            node_colors1[node] = colors1[comp_idx]
    
    nx.draw(G1, pos1, node_color=node_colors1, 
           with_labels=True, node_size=500, 
           edgecolors='black', linewidths=1)
    plt.title(f'Graph 1 ({len(components1)} components)')
    
    # Plot second graph
    plt.subplot(122)
    pos2 = nx.spring_layout(G2, seed=42)
    
    # Color nodes based on connected components for G2
    components2 = list(nx.connected_components(G2))
    colors2 = [f'C{i}' for i in range(len(components2))]
    node_colors2 = ['white'] * G2.number_of_nodes()
    for comp_idx, component in enumerate(components2):
        for node in component:
            node_colors2[node] = colors2[comp_idx]
    
    nx.draw(G2, pos2, node_color=node_colors2, 
           with_labels=True, node_size=500, 
           edgecolors='black', linewidths=1)
    plt.title(f'Graph 2 ({len(components2)} components)')
    
    # Add overall title
    plt.suptitle(f'Disconnected Pair {pair_idx} ({category})', y=1.05)
    
    # Save the plot
    plot_dir = "plots/disconnected_pairs"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/pair_{pair_idx}_{category.lower()}.png", 
                bbox_inches='tight', dpi=300)
    plt.close()


# create a function that runs through every pair in BREC, and
# check wehther the graphs are connected.
# keep count of connected and disconnected pairs by category
def check_connectivity_stats(dataset: BRECDataset) -> dict:
    """Check connectivity of all graph pairs in BREC dataset."""
    connectivity_stats : dict= {}
        
    # Initialize statistics
    for category in part_dict:
        connectivity_stats[category] = {
            'connected': 0,
            'disconnected': 0,
            'total_pairs': 0
        }
    
    # Check each pair in each category
    for category, (start, end) in part_dict.items():
        print(f"\nChecking connectivity for {category} category...")
        for pair_idx in range(start, end):
            # Get both graphs in the pair
            G1 = to_networkx(dataset[pair_idx * 2], to_undirected=True)
            G2 = to_networkx(dataset[pair_idx * 2 + 1], to_undirected=True)
            
            # Check connectivity
            is_G1_connected = nx.is_connected(G1)
            is_G2_connected = nx.is_connected(G2)
            
            connectivity_stats[category]['total_pairs'] += 1
            
            # If both graphs are connected
            if is_G1_connected and is_G2_connected:
                connectivity_stats[category]['connected'] += 1
            else:
                connectivity_stats[category]['disconnected'] += 1
                # Print which graphs are disconnected and plot them
                if not is_G1_connected and not is_G2_connected:
                    print(f"  Pair {pair_idx}: Both graphs are disconnected")
                elif not is_G1_connected:
                    print(f"  Pair {pair_idx}: First graph is disconnected")
                else:
                    print(f"  Pair {pair_idx}: Second graph is disconnected")
                
                # Plot the disconnected pair
                plot_disconnected_pair(G1, G2, pair_idx, category)
                print(f"  Plot saved to: plots/disconnected_pairs/pair_{pair_idx}_{category.lower()}.png")
    
    # Print summary
    print("\nConnectivity Summary:")
    print("-" * 60)
    print(f"{'Category':<20} {'Connected':<12} {'Disconnected':<12} {'Total':<12}")
    print("-" * 60)
    
    for category in connectivity_stats:
        stats = connectivity_stats[category]
        connected_percent = (stats['connected'] / stats['total_pairs']) * 100
        print(f"{category:<20} {stats['connected']:<12} {stats['disconnected']:<12} {stats['total_pairs']:<12} ({connected_percent:.1f}% connected)")
    
    return connectivity_stats

def process_pair(dataset: BRECDataset, encoding: str, pair_info: tuple) -> dict:
    """Process a single pair of graphs."""
    category, pair_idx = pair_info
    print(f"\nDEBUG: Processing pair_info: {pair_info}")
    print(f"DEBUG: Category: {category}, Pair Index: {pair_idx}")
    print(f"DEBUG: Dataset indices: {pair_idx * 2} and {pair_idx * 2 + 1}")
    
    # Get both graphs and check connectivity
    G1 = to_networkx(dataset[pair_idx * 2], to_undirected=True)
    G2 = to_networkx(dataset[pair_idx * 2 + 1], to_undirected=True)
    
    # Check regularity for Regular category
    if category == "Regular":
        is_reg1, deg1 = is_regular(G1)
        is_reg2, deg2 = is_regular(G2)
        assert is_reg1, f"Graph 1 in Regular pair {pair_idx} is not regular! Degrees: {[d for _, d in G1.degree()]}"
        assert is_reg2, f"Graph 2 in Regular pair {pair_idx} is not regular! Degrees: {[d for _, d in G2.degree()]}"
        print(f"Regular graphs confirmed: degrees {deg1} and {deg2}")
    
    # Skip if either graph is disconnected
    if not (nx.is_connected(G1) and nx.is_connected(G2)):
        print(f"Skipping pair {pair_idx} ({category}): Contains disconnected graph(s)")
        return None
    
    # Analyze connected pair
    pair_results = analyze_graph_pair(
        dataset[pair_idx * 2],
        dataset[pair_idx * 2 + 1],
        pair_idx,
        category,
        is_isomorphic=False,
        encoding=encoding,
    )
    
    return pair_results

@click.command()
@click.option(
    '--encoding',
    type=str,
    default="LAPE-Normalized",
    required=True,
    help='Encoding to analyze (e.g., "LAPE-Normalized", "LDP", etc.)'
)
@click.option(
    '--test-mode',
    is_flag=True, # default is False
    help='Run only 3 test pairs instead of full dataset'
)
@click.option(
    '--num-workers',
    type=int,
    default=mp.cpu_count() - 2,
    help='Number of worker processes to use'
)
def main(encoding: str, test_mode: bool, num_workers: int) -> None:
    """Analyze BREC dataset with specified encoding."""
    # Validate encoding
    valid_encodings = [enc[0] for enc in ENCODINGS_TO_CHECK]
    if encoding not in valid_encodings:
        raise click.BadParameter(
            f"Invalid encoding. Choose from: {', '.join(valid_encodings)}"
        )
    
    print(f"\nAnalyzing BREC dataset with {encoding} encoding...")
    print(f"Using {num_workers} worker processes")
    
    create_output_dirs()
    dataset : BRECDataset = BRECDataset()
    
    # First check connectivity
    print("\nAnalyzing connectivity of BREC dataset...")
    connectivity_stats : dict= check_connectivity_stats(dataset)
    
    # Save connectivity stats to JSON
    connectivity_file : str= "results/connectivity_stats.json"
    with open(connectivity_file, 'w') as f:
        json.dump(connectivity_stats, f, indent=2)
    print(f"\nConnectivity statistics saved to: {connectivity_file}")
    
    # Create results files with encoding-specific names
    results_file : str= f"results/comparisons_{encoding}.txt"
    json_file : str= f"results/statistics_{encoding}.json"
    
    # Initialize JSON results
    json_results : dict= {}
    
    with open(results_file, "w") as f:
        if test_mode:
            pairs_to_process : list[tuple[str, int]] = [
                ("Basic", 0),
                ("Basic", 1),
                ("Basic", 2)
            ]
        else:
            # Full analysis
            pairs_to_process : list[tuple[str, int]] = [
                (category, idx)
                for category, (start, end) in part_dict.items()
                for idx in range(start, end)
            ]

        print("Pairs to process:", pairs_to_process)
        
        # Print detailed information about pairs to process
        print("\nDEBUG: Detailed pairs to process:")
        for category, idx in pairs_to_process:
            print(f"Category: {category}, Index: {idx}, Dataset indices: {idx * 2} and {idx * 2 + 1}")
        
        print(f"\nDEBUG: Total pairs to process: {len(pairs_to_process)}")
        
        # Create a pool of worker processes
        with mp.Pool(num_workers) as pool:
            # Create a partial function with fixed arguments
            process_func = partial(process_pair, dataset, encoding)
            
            # Process pairs in parallel and collect results
            results : list[dict] = []
            for result in pool.imap_unordered(process_func, pairs_to_process):
                if result is not None:  # Skip None results (disconnected pairs)
                    results.append(result)
            
        # Write all results after parallel processing is complete
        for result in results:
            write_results(f, result, json_results)
        
        # Save JSON results with percentages
        final_stats : dict= {}
        for category in json_results:
            final_stats[category] = {}
            for enc in json_results[category]:
                stats = json_results[category][enc]
                percentage = (stats['different'] / stats['total']) * 100 if stats['total'] > 0 else 0
                final_stats[category][enc] = round(percentage, 2)
        
        # Save JSON file
        with open(json_file, 'w') as json_f:
            json.dump(final_stats, json_f, indent=2)
        
        print(f"\nStatistics saved to: {json_file}")
        
        # Generate LaTeX table
        generate_latex_table(final_stats)

def generate_latex_table(stats: dict) -> None:
    """Generate LaTeX table from statistics."""
    categories : list[str] = ["Basic", "Regular", "Extension", "CFI", "4-Vertex_Condition"]
    encodings : list[str] = [
        "Graph (1-WL)",
        "Hypergraph (1-WL)",
        "Graph (LDP)",
        "Hypergraph (LDP)",
        "Graph (LCP-FRC)",
        "Hypergraph (LCP-FRC)",
        "Graph (EE RWPE)",
        "Graph (EN RWPE)",
        "Graph (Hodge LAPE)",
        "Graph (Normalized LAPE)"
    ]
    
    latex_file : str = "results/comparison_table.tex"
    with open(latex_file, 'w') as f:
        f.write("\\begin{table*}[h!]\n\\centering\n\\tiny\n")
        f.write("\\begin{tabular}{|l|" + "c|"*len(categories) + "}\n\\hline\n")
        
        # Header
        f.write("\\textbf{Level (Encodings)} & " + 
                " & ".join([f"\\textbf{{{cat}}}" for cat in categories]) + 
                " \\\\\n\\hline\n")
        
        # Data rows
        for encoding in encodings:
            row = [encoding]
            for category in categories:
                value = stats.get(category, {}).get(encoding, "")
                row.append(f"{value}\\%" if value != "" else "")
            f.write(" & ".join(row) + " \\\\\n")
        
        # Table footer
        f.write("\\hline\n\\end{tabular}\n")
        f.write("\\caption{Difference in encodings on BREC dataset. We report the percentage of pairs with different encoding, at different level (graph or hypergraph)}\n")
        f.write("\\end{table*}\n")
        
    print(f"\nLaTeX table saved to: {latex_file}")

def is_regular(G: nx.Graph) -> tuple[bool, int]:
    """Check if a graph is regular (all nodes have same degree).
    
    Args:
        G: NetworkX graph
        
    Returns:
        tuple[bool, int]: (is_regular, degree if regular else -1)
    """
    if len(G) == 0:  # Empty graph
        return True, 0
        
    degrees = [d for _, d in G.degree()]
    first_degree = degrees[0]
    
    # Check if all degrees are equal to the first degree
    is_reg = all(d == first_degree for d in degrees)
    
    return is_reg, first_degree if is_reg else -1

if __name__ == "__main__":
    # Required for Windows compatibility
    mp.freeze_support()
    main()
