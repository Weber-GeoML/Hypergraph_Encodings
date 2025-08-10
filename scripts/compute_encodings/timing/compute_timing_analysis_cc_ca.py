#!/usr/bin/env python3
"""Timing analysis for coauthorship (CA) and cocitation (CC) datasets encoding computations.

This script loads pre-existing hypergraph datasets (CA and CC) and times only
the computation of hypergraph encodings (no lifting required as datasets are
already in hypergraph format).
"""

import argparse
import os
import pickle
import sys
import warnings
from typing import Any, Dict, List, Optional

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "..", "src"))
# Add current directory for timing_utils
sys.path.append(current_dir)

from encodings_hnns.encodings import HypergraphEncodings
from timing_for_encodings.timing_utils import TimingCollector, extract_hypergraph_stats
from timing_for_encodings.timing_utils2 import time_single_encoding_run

warnings.simplefilter("ignore")


def load_dataset(dataset_name: str, data_type: str) -> Dict[str, Any]:
    """Load a dataset from the data directory.

    Args:
        dataset_name: Name of the dataset (e.g., 'cora', 'dblp')
        data_type: Either 'coauthorship' or 'cocitation'

    Returns:
        Dictionary containing dataset components

    Raises:
        FileNotFoundError: If dataset files are not found
    """
    print(f"Loading {data_type} dataset: {dataset_name}")

    # Get project root and dataset path - FIXED: Added one more dirname
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    dataset_path = os.path.join(project_root, "data", data_type, dataset_name)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Load features (sparse matrix)
    features_path = os.path.join(dataset_path, "features.pickle")
    with open(features_path, "rb") as handle:
        features = pickle.load(handle)

    # Load hypergraph
    hypergraph_path = os.path.join(dataset_path, "hypergraph.pickle")
    with open(hypergraph_path, "rb") as handle:
        hypergraph = pickle.load(handle)

    # Load labels
    labels_path = os.path.join(dataset_path, "labels.pickle")
    with open(labels_path, "rb") as handle:
        labels = pickle.load(handle)

    # Convert sparse features to dense if needed
    if hasattr(features, "toarray"):
        features = features.toarray()

    # Create dataset dict
    dataset = {
        "hypergraph": hypergraph,
        "features": features,
        "labels": labels,
        "n": features.shape[0],
    }

    print(
        f"  ✓ Loaded dataset with {dataset['n']} nodes, "
        f"{len(hypergraph)} hyperedges, "
        f"{features.shape[1]} features"
    )

    return dataset


def time_single_encoding_run(
    dataset: Dict[str, Any],
    encoding_type: str,
    timing_collector: TimingCollector,
    dataset_name: str,
    **encoding_params,
) -> None:
    """Time a single encoding computation run.

    Args:
        dataset: Dataset dictionary
        encoding_type: Type of encoding to compute
        timing_collector: TimingCollector instance
        dataset_name: Name of the dataset
        **encoding_params: Additional parameters for encoding computation
    """
    # Create fresh copy of dataset for each run
    dataset_copy = {
        "hypergraph": dataset["hypergraph"].copy(),
        "features": dataset["features"].copy(),
        "labels": dataset["labels"].copy(),
        "n": dataset["n"],
    }

    # Extract hypergraph statistics for timing context
    hypergraph_stats = extract_hypergraph_stats(dataset_copy, dataset_name)

    # Initialize HypergraphEncodings
    hgencodings = HypergraphEncodings()

    # Time the specific encoding computation
    with timing_collector.time_encoding(encoding_type, hypergraph_stats):
        try:
            if encoding_type == "degree":
                hgencodings.add_degree_encodings(
                    dataset_copy,
                    verbose=False,
                    normalized=encoding_params.get("normalized", True),
                    dataset_name=None,  # Don't save during timing
                )
            elif encoding_type.startswith("random_walk_"):
                rw_type = encoding_type.split("_")[-1]  # Extract EE, EN, WE
                hgencodings.add_randowm_walks_encodings(  # FIXED: typo in method name
                    dataset_copy,
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
                    dataset_copy,
                    laplacian_type=laplacian_type,
                    normalized=encoding_params.get("normalized", True),
                    dataset_name=None,
                )
            elif encoding_type.startswith("curvature_"):
                curvature_type = encoding_type.split("_")[-1]  # Extract ORC, FRC
                hgencodings.add_curvature_encodings(
                    dataset_copy,
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


def time_dataset_encodings(
    dataset_name: str,
    data_type: str,
    num_runs: int = 5,
    encoding_types: Optional[List[str]] = None,
) -> TimingCollector:
    """Time all encoding computations for a single dataset."""
    if encoding_types is None:
        encoding_types = [
            # Skip degree for now due to dimension issues
            "degree",
            "curvature_ORC",
            "curvature_FRC",
            "laplacian_Hodge",
            "laplacian_Normalized",
            # "random_walk_EE",
            "random_walk_EN",
            # "random_walk_WE",
            # Skip curvature_ORC as it's very slow
        ]

    print(f"\n{'='*60}")
    print(f"TIMING ANALYSIS: {data_type}_{dataset_name}")
    print(f"{'='*60}")
    print(f"Number of runs per encoding: {num_runs}")
    print(f"Encoding types: {', '.join(encoding_types)}")

    # Load dataset once
    try:
        dataset = load_dataset(dataset_name, data_type)
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        return TimingCollector()

    # Initialize timing collector
    timing_collector = TimingCollector()

    # Time each encoding type
    for encoding_type in encoding_types:
        print(f"\n📊 Timing {encoding_type}...")

        successful_runs = 0
        failed_runs = 0

        for run_num in range(num_runs):
            print(f"  Run {run_num + 1}/{num_runs}...", end=" ", flush=True)

            try:
                time_single_encoding_run(
                    dataset=dataset,
                    encoding_type=encoding_type,
                    timing_collector=timing_collector,
                    dataset_name=f"{data_type}_{dataset_name}",
                )
                print("✓")
                successful_runs += 1
            except Exception as e:
                print(f"✗ Error: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}")
                import traceback

                traceback.print_exc()
                assert False
                failed_runs += 1

        # Print summary for this encoding type
        if successful_runs > 0:
            print(f"  → {successful_runs}/{num_runs} runs successful")
        else:
            print(f"  → All {failed_runs} runs failed - no timing data collected")

    return timing_collector


def time_all_datasets(
    num_runs: int = 5, encoding_types: Optional[List[str]] = None
) -> Dict[str, TimingCollector]:
    """Time encoding computations for all CC and CA datasets.

    Args:
        num_runs: Number of timing runs per encoding type
        encoding_types: List of encoding types to test (None for all)

    Returns:
        Dictionary mapping dataset names to their TimingCollector instances
    """
    print("🚀 STARTING COMPREHENSIVE TIMING ANALYSIS")
    print("=" * 80)

    # Define datasets to analyze
    datasets_to_analyze = [
        ("cora", "coauthorship"),
        ("dblp", "coauthorship"),
        ("citeseer", "cocitation"),
        ("cora", "cocitation"),
        ("pubmed", "cocitation"),
    ]

    timing_results = {}

    for dataset_name, data_type in datasets_to_analyze:
        dataset_key = f"{data_type}_{dataset_name}"

        try:
            timing_collector = time_dataset_encodings(
                dataset_name=dataset_name,
                data_type=data_type,
                num_runs=num_runs,
                encoding_types=encoding_types,
            )
            timing_results[dataset_key] = timing_collector

            # Print brief summary for this dataset
            stats = timing_collector.compute_statistics()
            if stats:
                print(f"\n📋 Summary for {dataset_key}:")
                for enc_type, enc_stats in sorted(stats.items()):
                    print(
                        f"  {enc_type}: {enc_stats['mean']:.3f}s ± {enc_stats['std']:.3f}s"
                    )

        except Exception as e:
            print(f"❌ Error processing {dataset_key}: {e}")
            timing_results[dataset_key] = TimingCollector()

    return timing_results


def generate_comprehensive_report(timing_data: Dict[str, TimingCollector]) -> str:
    """Generate a comprehensive timing analysis report.

    Args:
        timing_data: Dictionary mapping dataset names to TimingCollector instances

    Returns:
        String containing detailed timing analysis report
    """
    report_lines = []
    report_lines.append("🏆 COMPREHENSIVE TIMING ANALYSIS REPORT")
    report_lines.append("=" * 80)

    # Combine all timing data for overall analysis
    combined_collector = TimingCollector()

    for dataset_name, collector in timing_data.items():
        for encoding_type, records in collector.timing_data.items():
            for record in records:
                combined_collector.add_timing_record(encoding_type, record)

    # Overall performance ranking
    overall_stats = combined_collector.compute_statistics()
    if overall_stats:
        report_lines.append("\n🎯 OVERALL PERFORMANCE RANKING (Mean Time)")
        report_lines.append("-" * 50)

        ranked_encodings = sorted(overall_stats.items(), key=lambda x: x[1]["mean"])
        for i, (encoding_type, stats) in enumerate(ranked_encodings, 1):
            report_lines.append(
                f"{i:2d}. {encoding_type:<20} "
                f"{stats['mean']:.4f}s ± {stats['std']:.4f}s "
                f"({stats['count']} measurements)"
            )

    # Dataset-specific analysis
    report_lines.append("\n📊 DATASET-SPECIFIC PERFORMANCE")
    report_lines.append("-" * 50)

    for dataset_name, collector in sorted(timing_data.items()):
        if not collector.timing_data:
            continue

        report_lines.append(f"\n📁 {dataset_name}:")

        dataset_stats = collector.compute_statistics()
        for encoding_type, stats in sorted(dataset_stats.items()):
            report_lines.append(
                f"   {encoding_type:<20} {stats['mean']:.4f}s ± {stats['std']:.4f}s"
            )

    # Add overall summary table
    report_lines.append("\n" + combined_collector.get_timing_summary_table())

    # Add detailed report from combined collector
    report_lines.append("\n" + combined_collector.generate_timing_report())

    return "\n".join(report_lines)


def save_timing_analysis_results(
    timing_data: Dict[str, TimingCollector],
    output_dir: str = "computed_encodings/timing_analysis",
) -> None:
    """Save timing analysis results to files.

    Args:
        timing_data: Dictionary mapping dataset names to TimingCollector instances
        output_dir: Directory to save results in
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save individual dataset timing data
    for dataset_name, collector in timing_data.items():
        filename = os.path.join(output_dir, f"{dataset_name}_timing_data.json")
        collector.save_timing_results(filename, format_type="json")
        print(f"💾 Saved timing data for {dataset_name}: {filename}")

    # Combine all data and save comprehensive results
    combined_collector = TimingCollector()
    for dataset_name, collector in timing_data.items():
        for encoding_type, records in collector.timing_data.items():
            for record in records:
                combined_collector.add_timing_record(encoding_type, record)

    # Save comprehensive results
    comprehensive_file = os.path.join(output_dir, "cc_ca_comprehensive_timing.json")
    combined_collector.save_timing_results(comprehensive_file, format_type="json")

    # Save comprehensive report
    report = generate_comprehensive_report(timing_data)
    report_file = os.path.join(output_dir, "cc_ca_timing_report.txt")
    with open(report_file, "w") as f:
        f.write(report)

    print(f"💾 Saved comprehensive timing analysis: {comprehensive_file}")
    print(f"📄 Saved detailed report: {report_file}")


def main() -> None:
    """Main function to run timing analysis."""
    parser = argparse.ArgumentParser(
        description="Timing analysis for hypergraph encoding computations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of timing runs per encoding type (default: 5)",
    )

    parser.add_argument(
        "--encoding-types",
        nargs="+",  # FIXED: was nargs="random_walk_EE"
        default=None,
        help="Specific encoding types to test (default: all available)",
    )

    parser.add_argument(
        "--lifting-method",
        type=str,
        choices=["none"],
        default="none",
        help="Lifting method - 'none' for CC/CA datasets (pre-existing hypergraphs)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="computed_encodings/timing_analysis",
        help="Output directory for timing results",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,  # choices: "cocitation_cora", "cocitation_dblp", "cocitation_citeseer", "cocitation_pubmed", "coauthorship_cora", "coauthorship_dblp"
        help="Specific dataset to test (format: 'datatype_datasetname', e.g., 'cocitation_cora')",
    )

    args = parser.parse_args()

    print("⏱️  HYPERGRAPH ENCODING TIMING ANALYSIS")
    print("📊 PRE-EXISTING HYPERGRAPHS - ENCODING COMPUTATIONS ONLY")
    print("=" * 70)
    print("Configuration:")
    print(f"  - Runs per encoding: {args.num_runs}")
    print(f"  - Lifting method: {args.lifting_method} (no lifting needed)")
    print("  - Timing: Encoding computations only")
    if args.encoding_types:
        print(f"  - Encoding types: {', '.join(args.encoding_types)}")
    else:
        print("  - Encoding types: all")
    print(f"  - Output directory: {args.output_dir}")
    if args.dataset:
        print(f"  - Target dataset: {args.dataset}")
    else:
        print("  - Target datasets: all available CC/CA datasets")

    try:
        if args.dataset:
            # Time single dataset
            data_type, dataset_name = args.dataset.split("_", 1)
            timing_collector = time_dataset_encodings(
                dataset_name=dataset_name,
                data_type=data_type,
                num_runs=args.num_runs,
                encoding_types=args.encoding_types,
            )
            timing_data = {args.dataset: timing_collector}
        else:
            # Time all datasets
            timing_data = time_all_datasets(
                num_runs=args.num_runs, encoding_types=args.encoding_types
            )

        # Generate and save results
        save_timing_analysis_results(timing_data, args.output_dir)

        # Print final summary
        report = generate_comprehensive_report(timing_data)
        print("\n" + report)

        print(f"\n🎉 Timing analysis completed! Results saved to: {args.output_dir}")

    except Exception as e:
        print(f"❌ Error during timing analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
