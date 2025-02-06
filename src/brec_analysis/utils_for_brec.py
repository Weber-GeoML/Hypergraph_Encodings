"""Utility functions for the BREC dataset"""

import os

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


def create_output_dirs() -> None:
    """Create output directories for plots and results"""
    dirs = [
        "plots/graph_pairs",
        "plots/hypergraphs_pairs",
        "plots/encodings",
        "results/brec",
        "results/brec/ran",
    ]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)


# Convert to PyG Data objects
def nx_to_pyg(G: nx.Graph) -> Data:
    """Convert NetworkX graph to PyG Data object.

    Args:
        G:
            the NetworkX graph to convert

    Returns:
        the PyG Data object
    """
    edge_index = torch.tensor(
        [[e[0] for e in G.edges()], [e[1] for e in G.edges()]], dtype=torch.long
    )
    x = torch.empty(
        (G.number_of_nodes(), 0), dtype=torch.float
    )  # Empty features
    y = torch.zeros(G.number_of_nodes(), dtype=torch.long)
    return Data(x=x, y=y, edge_index=edge_index, num_nodes=G.number_of_nodes())


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


def create_comparison_table(
    stats1: dict, stats2: dict
) -> tuple[list[str], list[str]]:
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


def create_comparison_result(
    is_direct_match: bool,
    is_scaled_match: bool,
    scaling_factor: float | None = None,
) -> dict:
    """Create a standardized comparison result dictionary

    Args:
        is_direct_match:
            whether the encodings are the same
        is_scaled_match:
            whether the encodings are the same up to scaling
        scaling_factor:
            the scaling factor if the encodings are the same up to scaling

    Returns:
        result:
            a dictionary with the comparison result
    """
    if is_direct_match:
        return {"status": "MATCH", "scaling_factor": 1.0}
    elif is_scaled_match:
        return {"status": "SCALED_MATCH", "scaling_factor": scaling_factor}
    return {"status": "NO_MATCH", "scaling_factor": None}


# Save matrices in pmatrix format
def matrix_to_pmatrix(matrix: np.ndarray) -> str:
    """Save matrices in pmatrix format

    Args:
        matrix:
            the matrix to save

    Returns:
        the matrix in pmatrix format
    """
    latex_str = "\\begin{pmatrix}\n"
    for row in matrix:
        latex_str += " & ".join([f"{x:.4f}" for x in row]) + " \\\\\n"
    latex_str += "\\end{pmatrix}"
    return latex_str
