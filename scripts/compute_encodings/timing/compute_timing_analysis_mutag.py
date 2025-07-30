#!/usr/bin/env python3
"""Timing analysis for MUTAG dataset encoding computations.

This script loads the MUTAG dataset, lifts graphs to hypergraphs using clique
expansions, and compares timing between graph encodings and hypergraph encodings.

Plan:
1. Compute the encodings on the original MUTAG graphs
2. Lift the graphs to hypergraphs using clique method
3. Save as mutag_lifted.pickle
4. Compute the encodings on the mutag_lifted.pickle
5. Time all those and do a timing report

# Note that this script should be good for: mutag, enzymesm proteins,
# imdb, collab, reddit.

from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx, to_dense_adj

import time
import tqdm
import torch
import numpy as np
import pandas as pd

# import TU datasets
mutag = list(TUDataset(root="data", name="MUTAG"))
enzymes = list(TUDataset(root="data", name="ENZYMES"))
proteins = list(TUDataset(root="data", name="PROTEINS"))
imdb = list(TUDataset(root="data", name="IMDB-BINARY"))
collab = list(TUDataset(root="data", name="COLLAB"))
reddit = list(TUDataset(root="data", name="REDDIT-BINARY"))
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset

# Add the timing_utils2 to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


# Define local fallback functions first
def convert_graph_to_hypergraph_lrgb(graph_data: Data) -> Dict[str, Any]:
    """Convert PyTorch Geometric graph to hypergraph format using LRGB method."""
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


try:
    from timing_utils2 import (
        convert_graph_to_hypergraph_clique,
        time_single_encoding_run,
        save_hypergraphs_to_pickle,
        load_hypergraphs_from_pickle,
        compute_graph_statistics,
        compute_hypergraph_statistics,
    )
    from timing_utils import TimingCollector
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the correct environment and directory.")
    # For now, import individually if timing_utils2 doesn't work
    try:
        sys.path.append(os.path.join(current_dir, "..", "..", "src"))
        sys.path.append(os.path.join(current_dir, ".."))
        from encodings_hnns.encodings import HypergraphEncodings
        from encodings_hnns.liftings_and_expansions import lift_to_hypergraph
        from timing_utils import TimingCollector, extract_hypergraph_stats

        # Define the functions locally if timing_utils2 import fails
        def convert_graph_to_hypergraph_clique(graph_data: Data) -> Dict[str, Any]:
            """Convert PyTorch Geometric graph to hypergraph format using clique method."""
            return lift_to_hypergraph(graph_data, verbose=False, already_in_nx=False)

        def save_hypergraphs_to_pickle(
            hypergraphs: List[Dict[str, Any]], file_path: str
        ) -> None:
            """Save hypergraphs to a pickle file."""
            import pickle

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                pickle.dump(hypergraphs, f)
            print(
                f"💾 Saved {len(hypergraphs)} hypergraphs to: {os.path.abspath(file_path)}"
            )

        def load_hypergraphs_from_pickle(file_path: str) -> List[Dict[str, Any]]:
            """Load hypergraphs from a pickle file."""
            import pickle

            with open(file_path, "rb") as f:
                hypergraphs = pickle.load(f)
            print(
                f"📂 Loaded {len(hypergraphs)} hypergraphs from: {os.path.abspath(file_path)}"
            )
            return hypergraphs

        def compute_graph_statistics(graph_data: Data) -> Dict[str, Any]:
            """Compute basic statistics for a PyTorch Geometric graph."""
            return {
                "num_nodes": graph_data.num_nodes,
                "num_edges": graph_data.num_edges,
                "avg_degree": (
                    (2 * graph_data.num_edges) / graph_data.num_nodes
                    if graph_data.num_nodes > 0
                    else 0
                ),
            }

        def compute_hypergraph_statistics(hypergraph: Dict[str, Any]) -> Dict[str, Any]:
            """Compute basic statistics for a hypergraph."""
            hyperedges = hypergraph["hypergraph"]
            edge_sizes = [len(edge) for edge in hyperedges.values()]
            return {
                "num_nodes": hypergraph["n"],
                "num_hyperedges": len(hyperedges),
                "avg_hyperedge_size": np.mean(edge_sizes) if edge_sizes else 0,
            }

        def time_single_encoding_run(
            hypergraph: Dict[str, Any],
            encoding_type: str,
            timing_collector: TimingCollector,
            graph_name: str,
            lifting_method: str = "",
            encoding_params: Optional[Dict[str, Any]] = None,
        ) -> None:
            """Time a single encoding computation on a hypergraph."""
            from copy import deepcopy

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
                        curvature_type = encoding_type.split("_")[
                            -1
                        ]  # Extract ORC, FRC
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

    except ImportError as e2:
        print(f"Secondary import error: {e2}")
        sys.exit(1)


def time_graph_encodings(
    graphs: List[Data],
    encoding_types: List[str],
    timing_collector: TimingCollector,
    max_graphs: int = 10,
) -> None:
    """Time encoding computations on original graphs.

    Args:
        graphs: List of PyTorch Geometric Data objects
        encoding_types: List of encoding types to test
        timing_collector: Collector for timing statistics
        max_graphs: Maximum number of graphs to process
    """
    print(f"\n📊 Timing encodings on original MUTAG graphs...")
    print(f"Processing {min(max_graphs, len(graphs))} graphs")

    # Note: All encoding functions require hypergraph format, even for "original" graphs
    # We convert graphs to simple hypergraphs using LRGB method (edges become size-2 hyperedges)
    # This represents "original graph" timing by using the simplest possible hypergraph conversion

    for i in range(min(max_graphs, len(graphs))):
        if i % 10 == 0:
            print(f"  Processing graph {i+1}/{min(max_graphs, len(graphs))}...")

        graph = graphs[i]

        # Convert to simple hypergraph (LRGB method - edges become size-2 hyperedges)
        hypergraph = convert_graph_to_hypergraph_lrgb(graph)

        for encoding_type in encoding_types:
            try:
                time_single_encoding_run(
                    hypergraph=hypergraph,
                    encoding_type=encoding_type,
                    timing_collector=timing_collector,
                    graph_name=f"graph_{i}",
                    lifting_method="lrgb_original",
                )
            except Exception as e:
                print(
                    f"    ✗ Error on graph {i}, encoding {encoding_type}: {str(e)[:50]}..."
                )


def time_hypergraph_encodings(
    hypergraphs: List[Dict[str, Any]],
    encoding_types: List[str],
    timing_collector: TimingCollector,
    max_hypergraphs: int = 10,
) -> None:
    """Time encoding computations on clique-lifted hypergraphs.

    Args:
        hypergraphs: List of hypergraph dictionaries
        encoding_types: List of encoding types to test
        timing_collector: Collector for timing statistics
        max_hypergraphs: Maximum number of hypergraphs to process
    """
    print(f"\n📊 Timing encodings on clique-lifted MUTAG hypergraphs...")
    print(f"Processing {min(max_hypergraphs, len(hypergraphs))} hypergraphs")

    for i in range(min(max_hypergraphs, len(hypergraphs))):
        if i % 10 == 0:
            print(
                f"  Processing hypergraph {i+1}/{min(max_hypergraphs, len(hypergraphs))}..."
            )

        hypergraph = hypergraphs[i]

        for encoding_type in encoding_types:
            try:
                time_single_encoding_run(
                    hypergraph=hypergraph,
                    encoding_type=encoding_type,
                    timing_collector=timing_collector,
                    graph_name=f"hypergraph_{i}",
                    lifting_method="clique_lifted",
                )
            except Exception as e:
                print(
                    f"    ✗ Error on hypergraph {i}, encoding {encoding_type}: {str(e)[:50]}..."
                )


def lift_mutag_to_hypergraphs(
    graphs: List[Data], max_graphs: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Lift MUTAG graphs to hypergraphs using clique expansions.

    Args:
        graphs: List of PyTorch Geometric Data objects
        max_graphs: Maximum number of graphs to process (None for all)

    Returns:
        List of hypergraph dictionaries
    """
    if max_graphs is None:
        max_graphs = len(graphs)

    print(f"\n🔄 Lifting {min(max_graphs, len(graphs))} MUTAG graphs to hypergraphs...")

    hypergraphs = []

    for i in range(min(max_graphs, len(graphs))):
        if i % 10 == 0:
            print(f"  Lifting graph {i+1}/{min(max_graphs, len(graphs))}...")

        try:
            graph = graphs[i]
            hypergraph = convert_graph_to_hypergraph_clique(graph)
            hypergraphs.append(hypergraph)
        except Exception as e:
            print(f"    ✗ Error lifting graph {i}: {str(e)[:50]}...")
            continue

    print(f"  ✓ Successfully lifted {len(hypergraphs)} graphs to hypergraphs")
    return hypergraphs


def generate_mutag_timing_report(timing_collector: TimingCollector) -> str:
    """Generate a comprehensive timing report for MUTAG analysis."""
    report = []
    report.append("🏆 MUTAG DATASET ENCODING TIMING ANALYSIS")
    report.append("📊 ORIGINAL GRAPHS vs CLIQUE-LIFTED HYPERGRAPHS")
    report.append("=" * 80)
    report.append("")

    # Get all timing data
    timing_data = {}
    for encoding_type, records in timing_collector.timing_data.items():
        timing_data[encoding_type] = [record["computation_time"] for record in records]

    if not timing_data:
        report.append("❌ No timing data collected!")
        return "\n".join(report)

    # Group times by encoding type and method - FIXED PARSING
    grouped_times: Dict[str, Dict[str, List[float]]] = {}

    for timing_key, times in timing_data.items():
        if not times:
            continue

        # More robust parsing that preserves encoding subtypes
        method = None
        base_encoding = None

        if "_lrgb_original_graph_" in timing_key:
            method = "lrgb_original"
            # Extract everything before _lrgb_original_graph_
            base_encoding = timing_key.split("_lrgb_original_graph_")[0]
        elif "_clique_lifted_graph_" in timing_key:
            method = "clique_lifted"
            # Extract everything before _clique_lifted_graph_
            base_encoding = timing_key.split("_clique_lifted_graph_")[0]
        elif "_lrgb_original_hypergraph_" in timing_key:
            method = "lrgb_original"
            base_encoding = timing_key.split("_lrgb_original_hypergraph_")[0]
        elif "_clique_lifted_hypergraph_" in timing_key:
            method = "clique_lifted"
            base_encoding = timing_key.split("_clique_lifted_hypergraph_")[0]
        else:
            continue

        if method is None or base_encoding is None:
            continue

        # Initialize nested dictionaries
        if base_encoding not in grouped_times:
            grouped_times[base_encoding] = {}
        if method not in grouped_times[base_encoding]:
            grouped_times[base_encoding][method] = []

        # Add all times for this encoding+method combination
        grouped_times[base_encoding][method].extend(times)

    # Compute statistics from grouped times
    encoding_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    method_stats: Dict[str, Dict[str, Dict[str, float]]] = {
        "lrgb_original": {},
        "clique_lifted": {},
    }

    for base_encoding, method_data in grouped_times.items():
        encoding_stats[base_encoding] = {}

        for method, all_times in method_data.items():
            if all_times:  # Only if we have times
                encoding_stats[base_encoding][method] = {
                    "mean": np.mean(all_times),
                    "std": np.std(all_times),
                    "min": np.min(all_times),
                    "max": np.max(all_times),
                    "count": len(all_times),
                }

                # Store for method-specific analysis
                if method in method_stats:
                    method_stats[method][base_encoding] = {
                        "mean": np.mean(all_times),
                        "std": np.std(all_times),
                    }

    # Overall performance ranking
    report.append("🎯 ENCODING PERFORMANCE RANKING (Mean Time)")
    report.append("-" * 50)

    all_encodings = []
    for encoding_type, method_data in encoding_stats.items():
        for method_name, stats in method_data.items():
            all_encodings.append(
                (f"{encoding_type}_{method_name}", stats["mean"], stats["std"])
            )

    all_encodings.sort(key=lambda x: x[1])

    for i, (name, mean_time, std_time) in enumerate(all_encodings, 1):
        report.append(f" {i:2d}. {name:<35} {mean_time:.4f}s ± {std_time:.4f}s")

    report.append("")

    # Method-specific performance
    report.append("📊 METHOD-SPECIFIC PERFORMANCE")
    report.append("-" * 50)

    for method_name in ["lrgb_original", "clique_lifted"]:
        if method_name in method_stats and method_stats[method_name]:
            method_display = (
                "Original Graphs (LRGB)"
                if method_name == "lrgb_original"
                else "Clique-Lifted Hypergraphs"
            )
            report.append(f"\n📁 {method_display.upper()}:")

            # Sort by mean time
            sorted_encodings = sorted(
                method_stats[method_name].items(), key=lambda x: x[1]["mean"]
            )

            for encoding_type, stats in sorted_encodings:
                report.append(
                    f"   {encoding_type:<25} {stats['mean']:.4f}s ± {stats['std']:.4f}s"
                )

    report.append("")

    # Detailed timing table
    report.append("📋 DETAILED TIMING TABLE")
    report.append("-" * 90)
    report.append(
        f"{'Encoding Type':<20} {'Method':<20} {'Mean±Std (s)':<15} {'Min (s)':<8} {'Max (s)':<8} {'Count':<6}"
    )
    report.append("-" * 90)

    for encoding_type in sorted(encoding_stats.keys()):
        for method_name in ["lrgb_original", "clique_lifted"]:
            if method_name in encoding_stats[encoding_type]:
                stats = encoding_stats[encoding_type][method_name]
                method_display = (
                    "Original (LRGB)"
                    if method_name == "lrgb_original"
                    else "Clique-Lifted"
                )
                report.append(
                    f"{encoding_type:<20} {method_display:<20} "
                    f"{stats['mean']:.3f}±{stats['std']:.3f}{'':>4} "
                    f"{stats['min']:.3f}{'':>5} {stats['max']:.3f}{'':>5} "
                    f"{stats['count']:<6}"
                )

    report.append("")

    # Original vs Lifted comparison
    report.append("🔄 ORIGINAL GRAPHS vs CLIQUE-LIFTED HYPERGRAPHS COMPARISON")
    report.append("-" * 60)

    for encoding_type in sorted(encoding_stats.keys()):
        if (
            "lrgb_original" in encoding_stats[encoding_type]
            and "clique_lifted" in encoding_stats[encoding_type]
        ):

            original_time = encoding_stats[encoding_type]["lrgb_original"]["mean"]
            lifted_time = encoding_stats[encoding_type]["clique_lifted"]["mean"]

            if original_time > 0:
                ratio = lifted_time / original_time
                faster = "Original" if ratio > 1 else "Clique-Lifted"
                report.append(f"\n🔹 {encoding_type.upper()}:")
                report.append(
                    f"   Original: {original_time:.4f}s, Clique-Lifted: {lifted_time:.4f}s"
                )
                report.append(f"   Ratio: {ratio:.2f} ({faster} is faster)")

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


def main() -> None:
    """Main function to run MUTAG timing analysis."""
    parser = argparse.ArgumentParser(
        description="Timing analysis for MUTAG dataset encoding computations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Time all encodings on 50 graphs
  python compute_timing_analysis_mutag.py --max-graphs 50

  # Time only specific encodings
  python compute_timing_analysis_mutag.py --encoding-types degree random_walk_EE

  # Use existing lifted hypergraphs
  python compute_timing_analysis_mutag.py --use-existing-lifted
        """,
    )

    parser.add_argument(
        "--max-graphs",
        type=int,
        default=10,
        help="Maximum number of graphs to process (default: 50)",
    )

    parser.add_argument(
        "--encoding-types",
        nargs="+",
        default=[
            "degree",
            "random_walk_EE",
            "random_walk_EN",
            "laplacian_Hodge",
            "curvature_ORC",
        ],
        help="Specific encoding types to test (default: subset of available)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="computed_encodings/timing_analysis_mutag",
        help="Output directory for timing results",
    )

    parser.add_argument(
        "--lifted-file",
        type=str,
        default="computed_encodings/mutag_lifted.pickle",
        help="Path to save/load lifted hypergraphs",
    )

    parser.add_argument(
        "--use-existing-lifted",
        action="store_true",
        help="Use existing lifted hypergraphs instead of recomputing",
    )

    args = parser.parse_args()

    print("⏱️  MUTAG DATASET TIMING ANALYSIS")
    print("📊 ORIGINAL GRAPHS vs CLIQUE-LIFTED HYPERGRAPHS")
    print("=" * 70)
    print("Configuration:")
    print(f"  - Maximum graphs: {args.max_graphs}")
    print(f"  - Encoding types: {', '.join(args.encoding_types)}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Lifted file: {args.lifted_file}")
    print(f"  - Use existing lifted: {args.use_existing_lifted}")
    print()

    try:
        # Initialize timing collector
        timing_collector = TimingCollector()

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Load MUTAG dataset
        print("📂 Loading MUTAG dataset...")
        mutag_dataset = list(TUDataset(root="data", name="MUTAG"))
        print(f"  ✓ Loaded {len(mutag_dataset)} graphs from MUTAG dataset")

        # Limit to max_graphs
        graphs_to_process = mutag_dataset[: args.max_graphs]
        print(f"  📊 Processing {len(graphs_to_process)} graphs")

        # Show sample graph statistics
        if graphs_to_process:
            sample_stats = compute_graph_statistics(graphs_to_process[0])
            print(f"  📈 Sample graph stats: {sample_stats}")

        print("\n🚀 STARTING MUTAG TIMING ANALYSIS")
        print("=" * 80)

        # Step 1: Time encodings on original graphs (using LRGB conversion)
        print("\n📊 PHASE 1: TIMING ORIGINAL GRAPHS")
        print("-" * 50)
        time_graph_encodings(
            graphs_to_process, args.encoding_types, timing_collector, args.max_graphs
        )

        # Step 2: Lift graphs to hypergraphs or load existing
        hypergraphs = []

        if args.use_existing_lifted and os.path.exists(args.lifted_file):
            print(f"\n📂 PHASE 2A: LOADING EXISTING LIFTED HYPERGRAPHS")
            print("-" * 50)
            hypergraphs = load_hypergraphs_from_pickle(args.lifted_file)
            # Limit to max_graphs
            hypergraphs = hypergraphs[: args.max_graphs]
        else:
            print(f"\n🔄 PHASE 2B: LIFTING GRAPHS TO HYPERGRAPHS")
            print("-" * 50)
            hypergraphs = lift_mutag_to_hypergraphs(graphs_to_process, args.max_graphs)

            # Save lifted hypergraphs
            save_hypergraphs_to_pickle(hypergraphs, args.lifted_file)

        # Show sample hypergraph statistics
        if hypergraphs:
            sample_hg_stats = compute_hypergraph_statistics(hypergraphs[0])
            print(f"  📈 Sample hypergraph stats: {sample_hg_stats}")

        # Step 3: Time encodings on clique-lifted hypergraphs
        print("\n📊 PHASE 3: TIMING CLIQUE-LIFTED HYPERGRAPHS")
        print("-" * 50)
        time_hypergraph_encodings(
            hypergraphs, args.encoding_types, timing_collector, args.max_graphs
        )

        # Save timing data
        json_filename = os.path.join(args.output_dir, "mutag_timing_data.json")
        timing_collector.save_timing_results(json_filename, format_type="json")

        # Generate and save report
        report = generate_mutag_timing_report(timing_collector)
        report_filename = os.path.join(args.output_dir, "mutag_timing_report.txt")

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
