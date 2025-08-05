#!/usr/bin/env python3
"""Timing analysis for peptidesstruct dataset encoding computations.

This script loads the peptidesstruct dataset (graphs), pre-lifts them to hypergraphs
using the chosen method (clique or LRGB), and then times only the computation of
hypergraph encodings (lifting time is excluded from measurements).
"""

import os
import sys
import pickle
import warnings
import argparse
import time
from typing import Dict, Any, List, Optional, Tuple

import torch
import numpy as np

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
# Add compute_encodings directory to path for timing_utils
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from encodings_hnns.encodings import HypergraphEncodings
from compute_encodings.encoding_saver_lrgb import EncodingsSaverLRGB
from encodings_hnns.liftings_and_expansions import lift_to_hypergraph
from src.timing.timing_utils import TimingCollector, extract_hypergraph_stats

warnings.simplefilter("ignore")


def load_peptidesstruct_data(
    data_path: str, subset: str = "train", max_graphs: Optional[int] = None
) -> List[Any]:
    """Load peptidesstruct data from PyTorch files.

    Args:
        data_path: Path to the peptidesstruct directory
        subset: Which subset to load ('train', 'val', 'test')
        max_graphs: Maximum number of graphs to load (None for all)

    Returns:
        List of graph data objects

    Raises:
        FileNotFoundError: If the data files are not found
    """
    print(f"Loading peptidesstruct {subset} data from {data_path}")

    data_file = os.path.join(data_path, f"{subset}.pt")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # Load the PyTorch data
    data = torch.load(data_file)

    if max_graphs is not None:
        data = data[:max_graphs]
        print(f"  ✓ Loaded {len(data)} graphs (limited to {max_graphs})")
    else:
        print(f"  ✓ Loaded {len(data)} graphs")

    return data


def convert_graph_to_hypergraph_lrgb(graph_data: Any) -> Dict[str, Any]:
    """Convert a single graph to hypergraph format using LRGB method (edge-based).

    Args:
        graph_data: Single graph data object from peptidesstruct

    Returns:
        Dictionary containing hypergraph data in standard format
    """
    return EncodingsSaverLRGB._convert_to_hypergraph_lrgb(graph_data)


def convert_graph_to_hypergraph_clique(graph_data: Any) -> Dict[str, Any]:
    """Convert a single graph to hypergraph format using clique-based lifting.

    Args:
        graph_data: Single graph data object from peptidesstruct

    Returns:
        Dictionary containing hypergraph data in standard format
    """
    # Convert the PyTorch Geometric data to PyG Data object format needed by lift_to_hypergraph
    X = graph_data[0]  # node features
    edge_attr = graph_data[1]  # edge features
    edge_index = graph_data[2]  # connectivity
    y = graph_data[3]  # labels

    # Create PyG Data object
    from torch_geometric.data import Data

    pyg_graph = Data(
        x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=X.shape[0]
    )

    # Use clique-based lifting
    return lift_to_hypergraph(pyg_graph, verbose=False)


def time_graph_to_hypergraph_conversion(
    graphs: List[Any], timing_collector: TimingCollector, num_graphs: int = 5
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Time the conversion of graphs to hypergraphs using both methods.

    Args:
        graphs: List of graph data objects
        timing_collector: TimingCollector instance
        num_graphs: Number of graphs to convert and time

    Returns:
        Tuple of (LRGB converted hypergraphs, Clique converted hypergraphs)
    """
    print(f"\n🔄 Timing graph-to-hypergraph conversion for {num_graphs} graphs...")

    lrgb_hypergraphs = []
    clique_hypergraphs = []

    # Time LRGB conversion
    print(f"\n  📊 LRGB (Edge-based) Lifting:")
    lrgb_times = []

    for i in range(min(num_graphs, len(graphs))):
        print(f"    Graph {i+1}/{num_graphs}...", end=" ", flush=True)

        start_time = time.perf_counter()
        try:
            hypergraph = convert_graph_to_hypergraph_lrgb(graphs[i])
            end_time = time.perf_counter()

            conversion_time = end_time - start_time
            lrgb_times.append(conversion_time)
            lrgb_hypergraphs.append(hypergraph)

            # Add timing record
            timing_record = {
                "computation_time": conversion_time,
                "start_time": start_time,
                "end_time": end_time,
                "graph_index": i,
                "num_nodes": hypergraph["n"],
                "num_hyperedges": len(hypergraph["hypergraph"]),
            }
            timing_collector.add_timing_record("lrgb_lifting", timing_record)

            print(f"✓ ({conversion_time:.4f}s)")

        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}...")
            continue

    # Time Clique conversion
    print(f"\n  🧠 Clique-based Lifting:")
    clique_times = []

    for i in range(min(num_graphs, len(graphs))):
        print(f"    Graph {i+1}/{num_graphs}...", end=" ", flush=True)

        start_time = time.perf_counter()
        try:
            hypergraph = convert_graph_to_hypergraph_clique(graphs[i])
            end_time = time.perf_counter()

            conversion_time = end_time - start_time
            clique_times.append(conversion_time)
            clique_hypergraphs.append(hypergraph)

            # Add timing record
            timing_record = {
                "computation_time": conversion_time,
                "start_time": start_time,
                "end_time": end_time,
                "graph_index": i,
                "num_nodes": hypergraph["n"],
                "num_hyperedges": len(hypergraph["hypergraph"]),
            }
            timing_collector.add_timing_record("clique_lifting", timing_record)

            print(f"✓ ({conversion_time:.4f}s)")

        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}...")
            continue

    # Print summary
    if lrgb_times:
        avg_lrgb = sum(lrgb_times) / len(lrgb_times)
        print(f"  → Average LRGB lifting time: {avg_lrgb:.4f}s")

    if clique_times:
        avg_clique = sum(clique_times) / len(clique_times)
        print(f"  → Average Clique lifting time: {avg_clique:.4f}s")

    if lrgb_times and clique_times:
        speedup = avg_clique / avg_lrgb if avg_lrgb > 0 else float("inf")
        print(f"  → Clique lifting is {speedup:.1f}x slower than LRGB lifting")

    return lrgb_hypergraphs, clique_hypergraphs


def time_single_encoding_on_hypergraph(
    hypergraph: Dict[str, Any],
    encoding_type: str,
    timing_collector: TimingCollector,
    graph_index: int,
    **encoding_params,
) -> None:
    """Time a single encoding computation on a hypergraph.

    Args:
        hypergraph: Hypergraph dictionary
        encoding_type: Type of encoding to compute
        timing_collector: TimingCollector instance
        graph_index: Index of the graph being processed
        **encoding_params: Additional parameters for encoding computation
    """
    # Create fresh copy of hypergraph for each run
    hypergraph_copy = {
        "hypergraph": hypergraph["hypergraph"].copy(),
        "features": hypergraph["features"].copy(),
        "labels": hypergraph["labels"].copy(),
        "n": hypergraph["n"],
    }

    # Extract hypergraph statistics for timing context
    hypergraph_stats = extract_hypergraph_stats(
        hypergraph_copy, f"peptidesstruct_graph_{graph_index}"
    )
    hypergraph_stats["graph_index"] = graph_index

    # Initialize HypergraphEncodings
    hgencodings = HypergraphEncodings()

    # Time the specific encoding computation
    with timing_collector.time_encoding(encoding_type, hypergraph_stats):
        try:
            # Extract base encoding type (remove suffixes like _lrgb, _clique, _with_lifting)
            base_encoding_type = encoding_type
            if encoding_type.endswith(("_lrgb", "_clique", "_with_lifting")):
                # Remove these specific suffixes but keep the core encoding type intact
                for suffix in [
                    "_lrgb_with_lifting",
                    "_clique_with_lifting",
                    "_lrgb",
                    "_clique",
                    "_with_lifting",
                ]:
                    if encoding_type.endswith(suffix):
                        base_encoding_type = encoding_type[: -len(suffix)]
                        break

            if base_encoding_type == "degree":
                hgencodings.add_degree_encodings(
                    hypergraph_copy,
                    verbose=False,
                    normalized=encoding_params.get("normalized", True),
                    dataset_name=None,  # Don't save during timing
                )
            elif base_encoding_type.startswith("random_walk_"):
                rw_type = base_encoding_type.split("_")[-1]  # Extract EE, EN, WE
                hgencodings.add_randowm_walks_encodings(
                    hypergraph_copy,
                    rw_type=rw_type,
                    k=encoding_params.get("k", 20),
                    normalized=encoding_params.get("normalized", True),
                    dataset_name=None,
                )
            elif base_encoding_type.startswith("laplacian_"):
                laplacian_type = base_encoding_type.split("_", 1)[
                    1
                ]  # Extract Hodge, Normalized
                hgencodings.add_laplacian_encodings(
                    hypergraph_copy,
                    laplacian_type=laplacian_type,
                    normalized=encoding_params.get("normalized", True),
                    dataset_name=None,
                )
            elif base_encoding_type.startswith("curvature_"):
                curvature_type = base_encoding_type.split("_")[-1]  # Extract ORC, FRC
                hgencodings.add_curvature_encodings(
                    hypergraph_copy,
                    curvature_type=curvature_type,
                    normalized=encoding_params.get("normalized", True),
                    dataset_name=None,
                )
            else:
                raise ValueError(
                    f"Unknown encoding type: {base_encoding_type} (from {encoding_type})"
                )
        except Exception as e:
            # Re-raise the exception to be caught by the outer try-catch
            # This ensures timing is still recorded even if the encoding fails
            raise e


def time_encodings_on_hypergraphs(
    hypergraphs: List[Dict[str, Any]],
    timing_collector: TimingCollector,
    encoding_types: List[str],
    num_graphs: int = 5,
    lifting_method: str = "unknown",
) -> None:
    """Time encoding computations on pre-converted hypergraphs.

    Args:
        hypergraphs: List of hypergraph dictionaries
        timing_collector: TimingCollector instance
        encoding_types: List of encoding types to test
        num_graphs: Number of graphs to process
    """
    print(f"\n📊 Timing encoding computations on {num_graphs} hypergraphs...")

    for encoding_type in encoding_types:
        print(f"\n  🔍 Timing {encoding_type}...")

        successful_runs = 0
        failed_runs = 0

        for i in range(min(num_graphs, len(hypergraphs))):
            print(f"    Graph {i+1}/{num_graphs}...", end=" ", flush=True)

            try:
                time_single_encoding_on_hypergraph(
                    hypergraph=hypergraphs[i],
                    encoding_type=f"{encoding_type}_{lifting_method}",
                    timing_collector=timing_collector,
                    graph_index=i,
                )
                print("✓")
                successful_runs += 1
            except Exception as e:
                print(f"✗ Error: {str(e)[:50]}...")
                failed_runs += 1

        # Print summary for this encoding type
        if successful_runs > 0:
            print(f"    → {successful_runs}/{num_graphs} runs successful")
        else:
            print(f"    → All {failed_runs} runs failed - no timing data collected")


def time_combined_lifting_and_encoding(
    graphs: List[Any],
    timing_collector: TimingCollector,
    encoding_types: List[str],
    num_graphs: int = 5,
    lifting_method: str = "lrgb",
) -> None:
    """Time the combined process of lifting graphs to hypergraphs and computing encodings.

    Args:
        graphs: List of graph data objects
        timing_collector: TimingCollector instance
        encoding_types: List of encoding types to test
        num_graphs: Number of graphs to process
    """
    print(f"\n🚀 Timing COMBINED lifting + encoding for {num_graphs} graphs...")

    for encoding_type in encoding_types:
        print(f"\n  🔍 Timing {encoding_type} (with lifting)...")

        successful_runs = 0
        failed_runs = 0

        for i in range(min(num_graphs, len(graphs))):
            print(f"    Graph {i+1}/{num_graphs}...", end=" ", flush=True)

            try:
                start_time = time.perf_counter()

                # Step 1: Convert graph to hypergraph using specified method
                if lifting_method == "lrgb":
                    hypergraph = convert_graph_to_hypergraph_lrgb(graphs[i])
                elif lifting_method == "clique":
                    hypergraph = convert_graph_to_hypergraph_clique(graphs[i])
                else:
                    raise ValueError(f"Unknown lifting method: {lifting_method}")

                # Step 2: Compute encoding
                time_single_encoding_on_hypergraph(
                    hypergraph=hypergraph,
                    encoding_type=f"{encoding_type}_{lifting_method}_with_lifting",  # Add suffix to distinguish
                    timing_collector=timing_collector,
                    graph_index=i,
                )

                end_time = time.perf_counter()
                total_time = end_time - start_time

                # Also record the total combined time
                timing_record = {
                    "computation_time": total_time,
                    "start_time": start_time,
                    "end_time": end_time,
                    "graph_index": i,
                    "num_nodes": hypergraph["n"],
                    "num_hyperedges": len(hypergraph["hypergraph"]),
                }
                timing_collector.add_timing_record(
                    f"{encoding_type}_{lifting_method}_total_with_lifting",
                    timing_record,
                )

                print("✓")
                successful_runs += 1
            except Exception as e:
                print(f"✗ Error: {str(e)[:50]}...")
                failed_runs += 1

        # Print summary for this encoding type
        if successful_runs > 0:
            print(f"    → {successful_runs}/{num_graphs} runs successful")
        else:
            print(f"    → All {failed_runs} runs failed - no timing data collected")


def run_comprehensive_timing_analysis(
    data_path: str,
    subsets: List[str] = ["train", "val", "test"],  # Default: process all subsets
    num_graphs: Optional[int] = None,  # Default: process all graphs
    encoding_types: Optional[List[str]] = None,
    max_graphs_to_load: Optional[int] = None,
    lifting_method: str = "clique",
) -> Dict[str, TimingCollector]:
    """Run comprehensive timing analysis on peptidesstruct dataset.

    For each graph individually:
    1) Pre-lift graphs to hypergraphs using chosen method (NOT TIMED)
    2) Time only the encoding computations on the pre-lifted hypergraphs

    Args:
        data_path: Path to peptidesstruct data directory
        subsets: Which data subsets to use (default: all three)
        num_graphs: Number of graphs to process for timing (None for all)
        encoding_types: List of encoding types to test (None for all)
        max_graphs_to_load: Maximum graphs to load from dataset
        lifting_method: Method for graph-to-hypergraph conversion ('clique' or 'lrgb')

    Returns:
        Dictionary mapping subset names to TimingCollector instances
    """
    if encoding_types is None:
        encoding_types = [
            "degree",
            "random_walk_EE",
            "random_walk_EN",
            "random_walk_WE",
            "laplacian_Hodge",
            "laplacian_Normalized",
            "curvature_ORC",
            "curvature_FRC",
        ]

    print("🚀 STARTING PEPTIDESSTRUCT TIMING ANALYSIS")
    print("📊 PRE-LIFTED HYPERGRAPHS - ENCODING COMPUTATIONS ONLY")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Data path: {data_path}")
    print(f"  - Subsets: {', '.join(subsets)}")
    print(f"  - Graphs to process: {'all' if num_graphs is None else num_graphs}")
    print(f"  - Lifting method: {lifting_method} (pre-computed)")
    print(f"  - Timing: Encoding computations only")
    print(f"  - Encoding types: {', '.join(encoding_types)}")

    # Store results for all subsets
    all_timing_results = {}

    for subset in subsets:
        print(f"\n{'='*80}")
        print(f"PROCESSING SUBSET: {subset.upper()}")
        print(f"{'='*80}")

        try:
            # Load graph data
            graphs = load_peptidesstruct_data(data_path, subset, max_graphs_to_load)

            # Analyze the first graph structure
            print("\n🔍 ANALYZING GRAPH STRUCTURE:")
            print("=" * 50)

            graph = graphs[0]
            print(f"Type of graph: {type(graph)}")
            print(f"Number of elements in graph tuple: {len(graph)}")

            for i, element in enumerate(graph):
                print(f"\nElement {i}:")
                print(f"  Type: {type(element)}")
                print(
                    f"  Shape: {element.shape if hasattr(element, 'shape') else 'N/A'}"
                )
                print(
                    f"  Data type: {element.dtype if hasattr(element, 'dtype') else 'N/A'}"
                )

                if hasattr(element, "shape") and len(element.shape) == 2:
                    print(f"  Dimensions: {element.shape[0]} x {element.shape[1]}")
                    print(f"  Sample values (first 5 rows, first 5 cols):")
                    print(f"    {element[:5, :5]}")
                elif hasattr(element, "shape") and len(element.shape) == 1:
                    print(f"  Length: {element.shape[0]}")
                    print(f"  Sample values (first 10): {element[:10]}")
                else:
                    print(f"  Content preview: {str(element)[:100]}...")

            # Identify what each element likely represents
            print(f"\n📋 INTERPRETATION:")
            print("=" * 50)
            print(
                "Based on the structure, this appears to be a PyTorch Geometric Data object with:"
            )
            print("  - Element 0: Edge index (source, target node pairs)")
            print("  - Element 1: Edge features (3-dimensional)")
            print("  - Element 2: Node features (2-dimensional)")
            print("  - Element 3: Graph-level features (11-dimensional)")

            # Extract key information
            edge_index = graph[0]
            edge_features = graph[1]
            node_features = graph[2]
            graph_features = graph[3]

            print(f"\n📊 GRAPH STATISTICS:")
            print("=" * 50)
            print(f"Number of nodes: {node_features.shape[1]}")
            print(f"Number of edges: {edge_index.shape[1]}")
            print(f"Node feature dimension: {node_features.shape[0]}")
            print(f"Edge feature dimension: {edge_features.shape[1]}")
            print(f"Graph feature dimension: {graph_features.shape[1]}")

            # Check for self-loops and isolated nodes
            unique_nodes = set(edge_index[0].tolist() + edge_index[1].tolist())
            print(f"Unique nodes in edges: {len(unique_nodes)}")
            print(f"Self-loops: {sum(edge_index[0] == edge_index[1]).item()}")

            # Node degree analysis
            from collections import Counter

            node_degrees = Counter(edge_index[0].tolist() + edge_index[1].tolist())
            print(
                f"Average node degree: {sum(node_degrees.values()) / len(node_degrees):.2f}"
            )
            print(f"Max node degree: {max(node_degrees.values())}")
            print(f"Min node degree: {min(node_degrees.values())}")

            print("\n" + "=" * 50)
            print("Analysis complete. Continuing with timing...")
            print("=" * 50)

            # Determine actual number of graphs to process
            actual_num_graphs = (
                len(graphs) if num_graphs is None else min(num_graphs, len(graphs))
            )

            print(f"Processing {actual_num_graphs} graphs from {subset} subset")

            # Initialize timing collector for this subset
            timing_collector = TimingCollector()

            # Phase 1: Pre-lift all graphs to hypergraphs (NOT TIMED)
            print(
                f"\n🔄 Pre-lifting graphs to hypergraphs using {lifting_method.upper()} method..."
            )

            hypergraphs = []
            for i in range(actual_num_graphs):
                print(
                    f"  Lifting graph {i+1}/{actual_num_graphs}...", end=" ", flush=True
                )

                if lifting_method == "lrgb":
                    hypergraph = convert_graph_to_hypergraph_lrgb(graphs[i])
                else:  # clique
                    hypergraph = convert_graph_to_hypergraph_clique(graphs[i])

                hypergraphs.append(hypergraph)
                print("✓")

            print(
                f"  → All {actual_num_graphs} graphs successfully lifted to hypergraphs"
            )

            # Phase 2: Time encoding computations only (on pre-lifted hypergraphs)
            print(
                f"\n📊 Timing encoding computations on {lifting_method.upper()}-lifted hypergraphs..."
            )
            time_encodings_on_hypergraphs(
                hypergraphs,
                timing_collector,
                encoding_types,
                actual_num_graphs,
                lifting_method,
            )

            # Store timing results for this subset
            all_timing_results[subset] = timing_collector

        except Exception as e:
            print(f"\n❌ Error processing {subset} subset: {e}")
            import traceback

            traceback.print_exc()
            continue

    return all_timing_results


def generate_peptidesstruct_timing_report(timing_data: TimingCollector) -> str:
    """Generate a detailed timing report for peptidesstruct analysis.

    Args:
        timing_data: TimingCollector with collected timing data

    Returns:
        String containing detailed timing analysis report
    """
    report_lines = []
    report_lines.append("🏆 PEPTIDESSTRUCT TIMING ANALYSIS REPORT")
    report_lines.append("=" * 80)

    # Compute statistics
    stats = timing_data.compute_statistics()

    if not stats:
        report_lines.append("No timing data collected.")
        return "\n".join(report_lines)

    # Separate different types of timings
    conversion_stats = {k: v for k, v in stats.items() if "conversion" in k}
    encoding_only_stats = {
        k: v
        for k, v in stats.items()
        if not ("conversion" in k or "with_lifting" in k or "total_with_lifting" in k)
    }
    combined_encoding_stats = {
        k: v for k, v in stats.items() if "with_lifting" in k and "total" not in k
    }
    total_combined_stats = {k: v for k, v in stats.items() if "total_with_lifting" in k}

    # Graph-to-hypergraph conversion timing
    if conversion_stats:
        report_lines.append("\n🔄 GRAPH-TO-HYPERGRAPH CONVERSION TIMING")
        report_lines.append("-" * 50)
        for conv_type, conv_stats in conversion_stats.items():
            report_lines.append(
                f"{conv_type:<35} {conv_stats['mean']:.4f}s ± {conv_stats['std']:.4f}s "
                f"({conv_stats['count']} graphs)"
            )

    # Encoding-only timing (on pre-converted hypergraphs)
    if encoding_only_stats:
        report_lines.append("\n📊 ENCODING-ONLY TIMING (on pre-converted hypergraphs)")
        report_lines.append("-" * 50)
        ranked_encodings = sorted(
            encoding_only_stats.items(), key=lambda x: x[1]["mean"]
        )
        for i, (encoding_type, stats_data) in enumerate(ranked_encodings, 1):
            report_lines.append(
                f"{i:2d}. {encoding_type:<25} {stats_data['mean']:.4f}s ± {stats_data['std']:.4f}s "
                f"({stats_data['count']} graphs)"
            )

    # Combined timing (lifting + encoding)
    if total_combined_stats:
        report_lines.append("\n🚀 TOTAL TIMING (graph-to-hypergraph + encoding)")
        report_lines.append("-" * 50)
        ranked_combined = sorted(
            total_combined_stats.items(), key=lambda x: x[1]["mean"]
        )
        for i, (encoding_type, stats_data) in enumerate(ranked_combined, 1):
            clean_name = encoding_type.replace("_total_with_lifting", "")
            report_lines.append(
                f"{i:2d}. {clean_name:<25} {stats_data['mean']:.4f}s ± {stats_data['std']:.4f}s "
                f"({stats_data['count']} graphs)"
            )

    # Comparison table
    if encoding_only_stats and total_combined_stats:
        report_lines.append("\n📋 TIMING COMPARISON TABLE")
        report_lines.append("-" * 80)
        report_lines.append(
            f"{'Encoding Type':<20} {'Encoding Only (s)':<18} {'Total w/ Lifting (s)':<20} {'Overhead (%)':<12}"
        )
        report_lines.append("-" * 80)

        for encoding_type in encoding_only_stats.keys():
            encoding_mean = encoding_only_stats[encoding_type]["mean"]
            total_key = f"{encoding_type}_total_with_lifting"
            if total_key in total_combined_stats:
                total_mean = total_combined_stats[total_key]["mean"]
                overhead_pct = (
                    ((total_mean - encoding_mean) / encoding_mean) * 100
                    if encoding_mean > 0
                    else 0
                )
                report_lines.append(
                    f"{encoding_type:<20} {encoding_mean:<18.4f} {total_mean:<20.4f} {overhead_pct:<12.1f}"
                )

    # Add detailed statistics table
    report_lines.append("\n" + timing_data.get_timing_summary_table())

    return "\n".join(report_lines)


def save_peptidesstruct_timing_results(
    timing_data: TimingCollector,
    output_dir: str = "computed_encodings/timing_analysis_peptidesstruct",
    subset: str = "train",
) -> None:
    """Save timing analysis results to files.

    Args:
        timing_data: TimingCollector with timing data
        output_dir: Directory to save results in
        subset: Data subset name for file naming
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save timing data as JSON
    json_filename = os.path.join(
        output_dir, f"peptidesstruct_{subset}_timing_data.json"
    )
    timing_data.save_timing_results(json_filename, format_type="json")

    # Save comprehensive report
    report = generate_peptidesstruct_timing_report(timing_data)
    report_filename = os.path.join(
        output_dir, f"peptidesstruct_{subset}_timing_report.txt"
    )
    with open(report_filename, "w") as f:
        f.write(report)

    print(f"💾 Saved timing data: {os.path.abspath(json_filename)}")
    print(f"📄 Saved detailed report: {os.path.abspath(report_filename)}")


def main() -> None:
    """Main function to run peptidesstruct timing analysis."""
    parser = argparse.ArgumentParser(
        description="Timing analysis for peptidesstruct dataset encoding computations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/peptidesstruct",
        help="Path to peptidesstruct data directory (default: data/peptidesstruct)",
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
        help="Which data subset to use (default: all)",
    )

    parser.add_argument(
        "--num-graphs",
        type=int,
        default=100,
        help="Number of graphs to process for timing (default: all)",
    )

    parser.add_argument(
        "--max-graphs-to-load",
        type=int,
        default=None,
        help="Maximum number of graphs to load from dataset (default: None)",
    )

    parser.add_argument(
        "--encoding-types",
        nargs="+",
        default=None,
        help="Specific encoding types to test (default: + is all)",
    )

    parser.add_argument(
        "--lifting-method",
        type=str,
        choices=["clique", "lrgb"],
        default="clique",
        help="Lifting method to use for graph-to-hypergraph conversion (default: clique)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="computed_encodings/timing_analysis_peptidesstruct",
        help="Output directory for timing results",
    )

    args = parser.parse_args()

    print("⏱️  PEPTIDESSTRUCT TIMING ANALYSIS")
    print("📊 PRE-LIFTED HYPERGRAPHS - ENCODING COMPUTATIONS ONLY")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Data path: {args.data_path}")
    print(f"  - Subset: {args.subset}")
    print(
        f"  - Graphs to process: {'all' if args.num_graphs is None else args.num_graphs}"
    )
    print(f"  - Max graphs to load: {args.max_graphs_to_load or 'all'}")
    print(f"  - Lifting method: {args.lifting_method} (pre-computed)")
    print(f"  - Timing: Encoding computations only")
    print(f"  - Output directory: {args.output_dir}")
    if args.encoding_types:
        print(f"  - Encoding types: {', '.join(args.encoding_types)}")
    else:
        print(f"  - Encoding types: all (8 types)")

    try:
        # Run timing analysis
        all_timing_results = run_comprehensive_timing_analysis(
            data_path=args.data_path,
            subsets=[args.subset] if args.subset != "all" else ["train", "val", "test"],
            num_graphs=args.num_graphs,
            encoding_types=args.encoding_types,
            max_graphs_to_load=args.max_graphs_to_load,
            lifting_method=args.lifting_method,
        )

        # Save results and generate reports for each subset
        for subset, timing_collector in all_timing_results.items():
            save_peptidesstruct_timing_results(
                timing_collector, args.output_dir, subset
            )

            # Print final report for this subset
            print(f"\n{'='*60}")
            print(f"REPORT FOR {subset.upper()} SUBSET")
            print(f"{'='*60}")
            report = generate_peptidesstruct_timing_report(timing_collector)
            print(report)

        print(
            f"\n🎉 Timing analysis completed! Results saved to: {os.path.abspath(args.output_dir)}"
        )

    except Exception as e:
        print(f"❌ Error during timing analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
