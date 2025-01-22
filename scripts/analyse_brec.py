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
from brec_analysis.encodings_to_check import ENCODINGS_TO_CHECK
from brec_analysis.match_status import MatchStatus
import json
from pathlib import Path

def analyze_graph_pair(
    data1: Data, data2: Data, pair_idx: int|str, category: str, is_isomorphic: bool
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
    final_results = {
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
        "hypergraph",
        node_mapping,
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
            f.write(f"\n=== {result['description']} ===\n")
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

def main() -> None:
    """Main function to analyse the BREC dataset"""
    create_output_dirs()
    dataset = BRECDataset()
    
    # Create results files
    results_file = "results/all_comparisons.txt"
    json_file = "results/statistics.json"
    
    # Initialize JSON results
    json_results = {}
    
    with open(results_file, "w") as f:
        # First analyze Rook and Shrikhande graphs
        print("\nAnalyzing Rook and Shrikhande graphs...")

        # Load the graphs
        rook = nx.read_graph6("rook_graph.g6")
        shrikhande = nx.read_graph6("shrikhande.g6")

        rook_data = nx_to_pyg(rook)
        shrikhande_data = nx_to_pyg(shrikhande)

        # Analyze as a special pair
        print("Analyzing Rook vs Shrikhande")
        special_results = analyze_graph_pair(
            rook_data,
            shrikhande_data,
            pair_idx="rook_vs_shrikhande",
            category="Special",
            is_isomorphic=False,
        )
        write_results(f, special_results, json_results)
        
        # Then continue with BREC dataset analysis
        part_dict: dict[str, tuple[int, int]] = {
            "Basic": (0, 60),
            "Regular": (60, 160),
            "Extension": (160, 260),
            "CFI": (260, 360),
            "4-Vertex_Condition": (360, 380),
            "Distance_Regular": (380, 400),
        }

        # Process pairs and collect results
        for category, (start, end) in part_dict.items():
            print(f"\nProcessing {category} category...")
            for pair_idx in range(start, end):
                pair_results = analyze_graph_pair(
                    dataset[pair_idx * 2],
                    dataset[pair_idx * 2 + 1],
                    pair_idx,
                    category,
                    is_isomorphic=False,
                )
                write_results(f, pair_results, json_results)
        
        # Save JSON results with percentages
        final_stats = {}
        for category in json_results:
            final_stats[category] = {}
            for encoding in json_results[category]:
                stats = json_results[category][encoding]
                percentage = (stats['different'] / stats['total']) * 100 if stats['total'] > 0 else 0
                final_stats[category][encoding] = round(percentage, 2)
        
        # Save JSON file
        with open(json_file, 'w') as json_f:
            json.dump(final_stats, json_f, indent=2)
        
        print(f"\nStatistics saved to: {json_file}")
        
        # Generate LaTeX table
        generate_latex_table(final_stats)

def generate_latex_table(stats: dict) -> None:
    """Generate LaTeX table from statistics."""
    categories = ["Basic", "Regular", "Extension", "CFI", "4-Vertex_Condition"]
    encodings = [
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
    
    latex_file = "results/comparison_table.tex"
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

if __name__ == "__main__":
    main()
