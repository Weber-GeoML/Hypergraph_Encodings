#!/usr/bin/env python3
"""Timing analysis for Rook and Shrikhande graphs encoding computations.

This script loads the Rook and Shrikhande graphs from .g6 format, lifts them
to hypergraphs using clique expansions, and then times only the computation
of hypergraph encodings (lifting time is excluded from measurements).
"""

import argparse
import os
import sys
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# Add the src directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..", "..", "..", "src")
sys.path.append(src_dir)
compute_encodings_dir = os.path.join(current_dir, "..")
sys.path.append(compute_encodings_dir)

try:
    from encodings_hnns.encodings import HypergraphEncodings
    from encodings_hnns.liftings_and_expansions import lift_to_hypergraph
    from timing_utils import TimingCollector, extract_hypergraph_stats
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the correct environment and directory.")
    sys.exit(1)


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


def time_all_encodings_on_graph(
    graph_name: str,
    graph_data: Data,
    encoding_types: List[str],
    timing_collector: TimingCollector,
) -> None:
    """Time all encoding computations on a single graph using clique-lifted hypergraph.

    Args:
        graph_name: Name of the graph
        graph_data: PyTorch Geometric Data object
        encoding_types: List of encoding types to test
        timing_collector: Collector for timing statistics
    """
    print(f"\n📊 Processing {graph_name.upper()} graph...")
    print(f"Graph stats: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")

    # Pre-lift to hypergraph using clique expansions (NOT TIMED)
    print("🔄 Pre-lifting to hypergraph using clique expansions...")

    clique_hypergraph = convert_graph_to_hypergraph_clique(graph_data)
    print(f"  🧠 Clique conversion completed - ready for encoding timing")

    # Count hyperedges for information
    num_hyperedges = len(clique_hypergraph["hypergraph"])
    print(
        f"  📊 Hypergraph: {clique_hypergraph['n']} nodes, {num_hyperedges} hyperedges"
    )

    # NOW START TIMING - only for encoding computations
    print(f"\n🔍 Timing encodings on clique-lifted {graph_name}...")

    for encoding_type in encoding_types:
        print(f"  ⏱️  Timing {encoding_type}...", end=" ", flush=True)

        try:
            time_single_encoding_run(
                hypergraph=clique_hypergraph,
                encoding_type=encoding_type,
                timing_collector=timing_collector,
                graph_name=graph_name,
                lifting_method="clique",
            )
            print("✓")
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}...")


def generate_rook_shrikhande_report(timing_collector: TimingCollector) -> str:
    """Generate a comprehensive timing report for Rook and Shrikhande graphs.

    Args:
        timing_collector: Collector with timing data

    Returns:
        Formatted report string
    """
    report = []
    report.append("🏆 ROOK & SHRIKHANDE GRAPHS ENCODING TIMING ANALYSIS")
    report.append("📊 CLIQUE-LIFTED HYPERGRAPHS - ENCODING COMPUTATIONS ONLY")
    report.append("=" * 80)
    report.append("")

    # Get all timing data
    timing_data = {}
    for encoding_type, records in timing_collector.timing_data.items():
        timing_data[encoding_type] = [record["computation_time"] for record in records]

    if not timing_data:
        report.append("❌ No timing data collected!")
        return "\n".join(report)

    # Organize data by encoding type and graph
    encoding_stats = {}
    graph_stats = {"rook": {}, "shrikhande": {}}

    for encoding_type, times in timing_data.items():
        if not times:
            continue

        # Parse encoding type: encoding_clique_graphname
        parts = encoding_type.split("_")
        graph_name = parts[-1]  # rook or shrikhande
        # Remove _clique_graphname to get base encoding
        base_encoding = "_".join(parts[:-2]) if len(parts) > 2 else parts[0]

        # Store in organized structure
        if base_encoding not in encoding_stats:
            encoding_stats[base_encoding] = {}
        encoding_stats[base_encoding][graph_name] = {
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
            "count": len(times),
        }

        # Store for graph-specific analysis
        if graph_name in graph_stats:
            graph_stats[graph_name][base_encoding] = {
                "mean": np.mean(times),
                "std": np.std(times),
            }

    # Overall performance ranking
    report.append("🎯 ENCODING PERFORMANCE RANKING (Mean Time)")
    report.append("-" * 50)

    all_encodings = []
    for encoding_type, graph_data in encoding_stats.items():
        for graph_name, stats in graph_data.items():
            all_encodings.append(
                (f"{encoding_type}_{graph_name}", stats["mean"], stats["std"])
            )

    all_encodings.sort(key=lambda x: x[1])

    for i, (name, mean_time, std_time) in enumerate(all_encodings, 1):
        report.append(f" {i:2d}. {name:<30} {mean_time:.4f}s ± {std_time:.4f}s")

    report.append("")

    # Graph-specific performance
    report.append("📊 GRAPH-SPECIFIC PERFORMANCE")
    report.append("-" * 50)

    for graph_name in ["rook", "shrikhande"]:
        if graph_name in graph_stats and graph_stats[graph_name]:
            report.append(f"\n📁 {graph_name.upper()} GRAPH:")

            # Sort by mean time
            sorted_encodings = sorted(
                graph_stats[graph_name].items(), key=lambda x: x[1]["mean"]
            )

            for encoding_type, stats in sorted_encodings:
                report.append(
                    f"   {encoding_type:<25} {stats['mean']:.4f}s ± {stats['std']:.4f}s"
                )

    report.append("")

    # Detailed timing table
    report.append("📋 DETAILED TIMING TABLE")
    report.append("-" * 80)
    report.append(
        f"{'Encoding Type':<25} {'Graph':<12} {'Mean±Std (s)':<15} {'Min (s)':<8} {'Max (s)':<8} {'Count':<6}"
    )
    report.append("-" * 80)

    for encoding_type in sorted(encoding_stats.keys()):
        for graph_name in ["rook", "shrikhande"]:
            if graph_name in encoding_stats[encoding_type]:
                stats = encoding_stats[encoding_type][graph_name]
                report.append(
                    f"{encoding_type:<25} {graph_name:<12} "
                    f"{stats['mean']:.3f}±{stats['std']:.3f}{'':>6} "
                    f"{stats['min']:.3f}{'':>5} {stats['max']:.3f}{'':>5} "
                    f"{stats['count']:<6}"
                )

    report.append("")

    # Graph comparison
    report.append("🔄 ROOK vs SHRIKHANDE COMPARISON")
    report.append("-" * 50)

    for encoding_type in sorted(encoding_stats.keys()):
        if (
            "rook" in encoding_stats[encoding_type]
            and "shrikhande" in encoding_stats[encoding_type]
        ):

            rook_time = encoding_stats[encoding_type]["rook"]["mean"]
            shrikhande_time = encoding_stats[encoding_type]["shrikhande"]["mean"]

            if rook_time > 0:
                ratio = shrikhande_time / rook_time
                faster = "Rook" if ratio > 1 else "Shrikhande"
                report.append(f"\n🔹 {encoding_type.upper()}:")
                report.append(
                    f"   Rook: {rook_time:.4f}s, Shrikhande: {shrikhande_time:.4f}s"
                )
                report.append(f"   Ratio: {ratio:.2f} ({faster} is faster)")

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


def main() -> None:
    """Main function to run Rook and Shrikhande timing analysis."""
    parser = argparse.ArgumentParser(
        description="Timing analysis for Rook and Shrikhande graphs encoding computations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Time all encodings on both graphs
  python compute_timing_analysis_rook_shrikhande.py
  
  # Time only specific encodings
  python compute_timing_analysis_rook_shrikhande.py --encoding-types degree random_walk_EE
  
  # Time with custom output directory
  python compute_timing_analysis_rook_shrikhande.py --output-dir my_results
        """,
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/Rook_Shrikhande",
        help="Path to Rook and Shrikhande data directory (default: data/Rook_Shrikhande)",
    )

    parser.add_argument(
        "--encoding-types",
        nargs="+",
        default=[
            "degree",
            "random_walk_EE",
            "random_walk_EN",
            "random_walk_WE",
            "laplacian_Hodge",
            "laplacian_Normalized",
            "curvature_ORC",
            "curvature_FRC",
        ],
        help="Specific encoding types to test (default: all available)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="computed_encodings/timing_analysis_rook_shrikhande",
        help="Output directory for timing results",
    )

    args = parser.parse_args()

    print("⏱️  ROOK & SHRIKHANDE TIMING ANALYSIS")
    print("📊 CLIQUE-LIFTED HYPERGRAPHS - ENCODING COMPUTATIONS ONLY")
    print("=" * 70)
    print("Configuration:")
    print(f"  - Data path: {args.data_path}")
    print(f"  - Encoding types: {', '.join(args.encoding_types)}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Lifting method: Clique expansions (pre-computed)")
    print(f"  - Timing: Encoding computations only")
    print()

    try:
        # Initialize timing collector
        timing_collector = TimingCollector()

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Load graphs
        rook_path = os.path.join(args.data_path, "rook_graph.g6")
        shrikhande_path = os.path.join(args.data_path, "shrikhande.g6")

        if not os.path.exists(rook_path):
            raise FileNotFoundError(f"Rook graph not found at: {rook_path}")
        if not os.path.exists(shrikhande_path):
            raise FileNotFoundError(f"Shrikhande graph not found at: {shrikhande_path}")

        rook_graph = load_g6_graph(rook_path)
        shrikhande_graph = load_g6_graph(shrikhande_path)

        print("\n🚀 STARTING ROOK & SHRIKHANDE TIMING ANALYSIS")
        print("=" * 80)

        # Time encodings on both graphs
        time_all_encodings_on_graph(
            "rook", rook_graph, args.encoding_types, timing_collector
        )

        time_all_encodings_on_graph(
            "shrikhande", shrikhande_graph, args.encoding_types, timing_collector
        )

        # Save timing data
        json_filename = os.path.join(
            args.output_dir, "rook_shrikhande_timing_data.json"
        )
        timing_collector.save_timing_results(json_filename, format_type="json")

        # Generate and save report
        report = generate_rook_shrikhande_report(timing_collector)
        report_filename = os.path.join(
            args.output_dir, "rook_shrikhande_timing_report.txt"
        )

        with open(report_filename, "w") as f:
            f.write(report)

        print(f"\n💾 Saved timing data: {os.path.abspath(json_filename)}")
        print(f"📄 Saved detailed report: {os.path.abspath(report_filename)}")

        # Print report
        print("\n" + report)

        print(
            f"\n🎉 Timing analysis completed! Results saved to: {os.path.abspath(args.output_dir)}"
        )

    except Exception as e:
        print(f"\n❌ Error during timing analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
