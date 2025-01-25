from encodings_hnns.orc_from_southern import ollivier_ricci_curvature, prob_rw, prob_two_hop
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
from scipy.stats import ks_2samp
from encodings_hnns.orc_from_southern import ollivier_ricci_curvature, prob_rw, prob_two_hop
import numpy as np
import os
from brec_analysis.analyse_brec_categories import analyze_brec_categories

def southern_orc_example(rook, shrikhande) -> None:
    # After loading rook and shrikhande graphs
    print("Computing ORCs for Rook and Shrikhande graphs...")

    # Compute ORCs with different probability measures and alpha values
    alpha_values = [0.0, 0.5]
    prob_measures = {
        "Default": None,
        "Random Walk": prob_rw,
        "Two Hop": prob_two_hop
    }

    for alpha in alpha_values:
        print(f"\nResults with alpha = {alpha}:")
        for measure_name, prob_fn in prob_measures.items():
            print(f"\n{measure_name} probability measure:")
            
            rook_orc = ollivier_ricci_curvature(rook, alpha=alpha, prob_fn=prob_fn)
            shrikhande_orc = ollivier_ricci_curvature(shrikhande, alpha=alpha, prob_fn=prob_fn)
            
            print("Rook Graph:")
            print(f"Mean curvature: {rook_orc.mean():.4f}")
            print(f"Min curvature: {rook_orc.min():.4f}")
            print(f"Max curvature: {rook_orc.max():.4f}")
            
            print("\nShrikhande Graph:")
            print(f"Mean curvature: {shrikhande_orc.mean():.4f}")
            print(f"Min curvature: {shrikhande_orc.min():.4f}")
            print(f"Max curvature: {shrikhande_orc.max():.4f}")
            
            # Check if distributions are different
            stat, pval = ks_2samp(rook_orc, shrikhande_orc)
            print(f"\nKS test p-value: {pval:.4f}")
            print(f"Distributions are {'different' if pval < 0.05 else 'similar'}")