#!/usr/bin/env python3
"""Shared timing utilities for graph and hypergraph encoding analysis.

This module contains common functions used across different timing analysis scripts
for converting graphs to hypergraphs and timing encoding computations.
"""

import os
import pickle
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from encodings_hnns.encodings import HypergraphEncodings
from encodings_hnns.liftings_and_expansions import lift_to_hypergraph
from timing_for_encodings.timing_utils import TimingCollector, extract_hypergraph_stats


def load_g6_graph(file_path: str) -> Data:
    """Load a graph from .g6 format and convert to PyTorch Geometric Data.

    Args:
        file_path: Path to the .g6 file

    Returns:
        PyTorch Geometric Data object
    """
    # Load graph from .g6 format using NetworkX
    G = nx.read_graph6(file_path)

    # Convert to PyTorch Geometric format
    data = from_networkx(G)

    # Add dummy node features (required for encodings)
    num_nodes = data.num_nodes
    data.x = torch.ones(num_nodes, 1, dtype=torch.float)  # Simple constant features

    # Ensure edge_index is of correct type
    data.edge_index = data.edge_index.long()

    print(f"Loaded graph from {file_path}: {num_nodes} nodes, {data.num_edges} edges")

    return data


def convert_graph_to_hypergraph_clique(graph_data: Data) -> Dict[str, Any]:
    """Convert PyTorch Geometric graph to hypergraph format using clique method.

    Args:
        graph_data: PyTorch Geometric Data object

    Returns:
        Dictionary in hypergraph format expected by encoding functions
    """
    # Use clique-based lifting directly on the PyTorch Geometric graph
    hypergraph = lift_to_hypergraph(graph_data, verbose=False, already_in_nx=False)
    return hypergraph


def convert_graph_to_hypergraph_lrgb(graph_data: Data) -> Dict[str, Any]:
    """Convert PyTorch Geometric graph to hypergraph format using LRGB method.

    Args:
        graph_data: PyTorch Geometric Data object

    Returns:
        Dictionary in hypergraph format expected by encoding functions
    """
    # Simple LRGB conversion: each edge becomes a size-2 hyperedge
    edge_index = graph_data.edge_index
    num_nodes = graph_data.num_nodes

    # Create hypergraph dictionary
    hypergraph_dict = {}
    edge_count = 0

    # Convert each edge to a hyperedge
    for i in range(edge_index.shape[1]):
        source, target = edge_index[0, i].item(), edge_index[1, i].item()
        if source != target:  # Skip self-loops
            hypergraph_dict[str(edge_count)] = [source, target]
            edge_count += 1

    # Create proper numpy arrays for features and labels
    features = np.ones((num_nodes, 1), dtype=np.float32)  # Simple constant features
    labels = np.zeros((num_nodes, 1), dtype=np.int32)  # Dummy labels

    hypergraph = {
        "hypergraph": hypergraph_dict,
        "n": num_nodes,
        "features": features,
        "labels": labels,
    }

    return hypergraph


def time_single_encoding_run(
    hypergraph: Dict[str, Any],
    encoding_type: str,
    timing_collector: TimingCollector,
    graph_name: str,
    lifting_method: str = "",
    encoding_params: Optional[Dict[str, Any]] = None,
) -> None:
    """Time a single encoding computation on a hypergraph.

    Args:
        hypergraph: Hypergraph dictionary
        encoding_type: Type of encoding to compute
        timing_collector: Collector for timing statistics
        graph_name: Name of the graph being processed
        lifting_method: Method used for lifting (for naming)
        encoding_params: Parameters for encoding computation
    """
    if encoding_params is None:
        encoding_params = {}

    # Create a copy to avoid modifying the original
    hypergraph_copy = deepcopy(hypergraph)

    # Extract hypergraph statistics for timing
    hypergraph_stats = extract_hypergraph_stats(hypergraph, graph_name)
    hypergraph_stats["lifting_method"] = lifting_method

    # Initialize HypergraphEncodings
    hgencodings = HypergraphEncodings()

    # Create timing key
    timing_key = f"{encoding_type}"
    if lifting_method:
        timing_key += f"_{lifting_method}"
    timing_key += f"_{graph_name}"

    # Time the specific encoding computation
    with timing_collector.time_encoding(timing_key, hypergraph_stats):
        try:
            if encoding_type == "degree":
                hgencodings.add_degree_encodings(
                    hypergraph_copy,
                    verbose=False,
                    normalized=encoding_params.get("normalized", True),
                    dataset_name=None,  # Don't save during timing
                )
            elif encoding_type.startswith("random_walk_"):
                rw_type = encoding_type.split("_")[-1]  # Extract EE, EN, WE
                hgencodings.add_randowm_walks_encodings(
                    hypergraph_copy,
                    rw_type=rw_type,
                    k=encoding_params.get("k", 20),
                    normalized=encoding_params.get("normalized", True),
                    dataset_name=None,
                )
            elif encoding_type.startswith("laplacian_"):
                laplacian_type = encoding_type.split("_", 1)[
                    1
                ]  # Extract Hodge, Normalized
                hgencodings.add_laplacian_encodings(
                    hypergraph_copy,
                    laplacian_type=laplacian_type,
                    normalized=encoding_params.get("normalized", True),
                    dataset_name=None,
                    use_same_sign=True,  # Use same sign for consistency
                )
            elif encoding_type.startswith("curvature_"):
                curvature_type = encoding_type.split("_")[-1]  # Extract ORC, FRC
                hgencodings.add_curvature_encodings(
                    hypergraph_copy,
                    curvature_type=curvature_type,
                    normalized=encoding_params.get("normalized", True),
                    dataset_name=None,
                )
            else:
                raise ValueError(f"Unknown encoding type: {encoding_type}")
        except Exception as e:
            # Re-raise the exception to be caught by the outer try-catch
            # This ensures timing is still recorded even if the encoding fails
            raise e


def save_hypergraphs_to_pickle(
    hypergraphs: List[Dict[str, Any]],
    file_path: str,
    lifting_times: Optional[List[float]] = None,
) -> None:
    """Save hypergraphs and optionally lifting times to a pickle file.

    Args:
        hypergraphs: List of hypergraph dictionaries
        file_path: Path to save the pickle file
        lifting_times: Optional list of lifting times in seconds
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save as a dictionary to support both hypergraphs and lifting_times
    data_to_save = {
        "hypergraphs": hypergraphs,
        "lifting_times": lifting_times if lifting_times is not None else [],
    }

    with open(file_path, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"💾 Saved {len(hypergraphs)} hypergraphs to: {os.path.abspath(file_path)}")


def load_hypergraphs_from_pickle(
    file_path: str,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Load hypergraphs and lifting times from a pickle file.

    Args:
        file_path: Path to the pickle file

    Returns:
        Tuple of (hypergraph dictionaries, lifting times)
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Handle backward compatibility
    if isinstance(data, list):
        # Old format: just hypergraphs
        hypergraphs = data
        lifting_times = []
        print(
            f"📂 Loaded {len(hypergraphs)} hypergraphs from: {os.path.abspath(file_path)} (no lifting times)"
        )
    else:
        # New format: dictionary with hypergraphs and lifting_times
        hypergraphs = data.get("hypergraphs", [])
        lifting_times = data.get("lifting_times", [])
        print(
            f"📂 Loaded {len(hypergraphs)} hypergraphs and {len(lifting_times)} lifting times from: {os.path.abspath(file_path)}"
        )

    return hypergraphs, lifting_times


def compute_graph_statistics(graph_data: Data) -> Dict[str, Any]:
    """Compute basic statistics for a PyTorch Geometric graph.

    Args:
        graph_data: PyTorch Geometric Data object

    Returns:
        Dictionary with graph statistics
    """
    return {
        "num_nodes": graph_data.num_nodes,
        "num_edges": graph_data.num_edges,
        "avg_degree": (
            (2 * graph_data.num_edges) / graph_data.num_nodes
            if graph_data.num_nodes > 0
            else 0
        ),
        "has_node_features": hasattr(graph_data, "x") and graph_data.x is not None,
        "has_edge_features": hasattr(graph_data, "edge_attr")
        and graph_data.edge_attr is not None,
    }


def compute_hypergraph_statistics(hypergraph: Dict[str, Any]) -> Dict[str, Any]:
    """Compute basic statistics for a hypergraph.

    Args:
        hypergraph: Hypergraph dictionary

    Returns:
        Dictionary with hypergraph statistics
    """
    hyperedges = hypergraph["hypergraph"]
    edge_sizes = [len(edge) for edge in hyperedges.values()]

    return {
        "num_nodes": hypergraph["n"],
        "num_hyperedges": len(hyperedges),
        "avg_hyperedge_size": np.mean(edge_sizes) if edge_sizes else 0,
        "min_hyperedge_size": min(edge_sizes) if edge_sizes else 0,
        "max_hyperedge_size": max(edge_sizes) if edge_sizes else 0,
        "total_incidences": sum(edge_sizes),
    }
