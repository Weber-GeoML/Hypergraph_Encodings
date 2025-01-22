"""
This script is used to analyse the BREC dataset.
It is used to compare the encodings of the graphs in the BREC dataset.
"""

import os

import hypernetx as hnx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from brec.dataset import BRECDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from encodings_hnns.liftings_and_expansions import lift_to_hypergraph
from encodings_hnns.encodings import HypergraphEncodings
from brec_analysis.check_encodings_same import (
    checks_encodings,
    find_isomorphism_mapping,
)
from brec_analysis.utils_for_brec import create_output_dirs, create_comparison_table, convert_nx_to_hypergraph_dict, nx_to_pyg
from brec_analysis.plotting_graphs_and_hgraphs_for_brec import plot_graph_pair, plot_hypergraph_pair
from brec_analysis.compare_encodings_wrapper import compare_encodings_wrapper

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

    # Compare graph-level encodings
    print(f"\nAnalyzing pair {pair_idx} ({category}):")
    print("\n")
    results = compare_encodings_wrapper(
        hg1, hg2, pair_idx, category, is_isomorphic, "graph", node_mapping
    )

    del hg1, hg2

    print("*-" * 25)
    print("*-" * 25)
    print(f"Analyzing pair {pair_idx} ({category}): at the hypergraph level")
    print("*-" * 25)
    print("*-" * 25)

    # Lift to hypergraphs
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
    results = compare_encodings_wrapper(
        hg1_lifted,
        hg2_lifted,
        pair_idx,
        category,
        is_isomorphic,
        "hypergraph",
        node_mapping,
    )

    return results

def main() -> None:
    """Main function to analyse the BREC dataset"""
    create_output_dirs()
    dataset = BRECDataset()
    
    # Create a single results file
    results_file = "results/all_comparisons.txt"
    
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
        write_results(f, special_results)
        
        # Then continue with BREC dataset analysis
        part_dict: dict[str, tuple[int, int]] = {
            "Basic": (0, 60),
            "Regular": (60, 160),
            "Extension": (160, 260),
            "CFI": (260, 360),
            "4-Vertex_Condition": (360, 380),
            "Distance_Regular": (380, 400),
        }

        # Process BREC dataset
        for category, (start, end) in part_dict.items():
            print(f"\nProcessing {category} category...")
            for pair_idx in range(start, end):
                results = analyze_graph_pair(
                    dataset[pair_idx * 2],
                    dataset[pair_idx * 2 + 1],
                    pair_idx,
                    category,
                    is_isomorphic=False,
                )
                write_results(f, results)

def write_results(f, results: dict) -> None:
    """Write results for a pair to the file."""
    f.write(f"\nAnalysis for pair {results['pair_idx']} ({results['category']}) - {results['level']} level\n")
    f.write(f"Isomorphic: {results['is_isomorphic']}\n\n")
    
    for encoding_type, result in results["encodings"].items():
        f.write(f"\n=== {result['description']} ===\n")
        f.write(f"Result: {'Same' if result['is_same'] else 'Different'}\n")

if __name__ == "__main__":
    main()
