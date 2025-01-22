"""Utility functions for the BREC dataset"""

import os

import networkx as nx
import numpy as np
import torch


def create_output_dirs() -> None:
    """Create output directories for plots and results"""
    dirs = [
        "plots/graph_pairs",
        "plots/hypergraphs",
        "plots/encodings",
        "results",
    ]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)


def convert_nx_to_hypergraph_dict(G: nx.Graph) -> dict:
    """Convert NetworkX graph to hypergraph dictionary format

    Args:
        G (nx.Graph):
            The NetworkX graph to convert.

    Returns:
        dict: The hypergraph dictionary.
    """
    hyperedges = {f"e_{i}": list(edge) for i, edge in enumerate(G.edges())}
    n = G.number_of_nodes()
    features = torch.empty((n, 0))
    return {
        "hypergraph": hyperedges,
        "features": features,
        "labels": {},
        "n": n,
    }


def create_comparison_table(stats1: dict, stats2: dict) -> tuple[list[str], list[str]]:
    """Create comparison table with differences highlighted in red.

    Args:
        stats1 (dict):
            Statistics for the first graph.
        stats2 (dict):
            Statistics for the second graph.

    Returns:
        tuple[list[str], list[str]]: A tuple containing the table text and colors.
    """
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
        colors.append("red" if is_different else "black")

    return table_text, colors


# Save matrices in pmatrix format
def matrix_to_pmatrix(matrix) -> str:
    latex_str = "\\begin{pmatrix}\n"
    for row in matrix:
        latex_str += " & ".join([f"{x:.4f}" for x in row]) + " \\\\\n"
    latex_str += "\\end{pmatrix}"
    return latex_str
