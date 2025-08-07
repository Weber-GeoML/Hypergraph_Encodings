#!/usr/bin/env python3
"""Timing analysis for hypergraph classification datasets encoding computations.

This script loads pre-existing hypergraph classification datasets and times only
the computation of hypergraph encodings (no lifting required as datasets are
already in hypergraph format).

mutag_hypergraphs.pickle - MUTAG dataset (75KB)
reddit_hypergraphs.pickle - Reddit dataset (17MB)
imdb_hypergraphs.pickle - IMDB dataset (245KB)
enzymes_hypergraphs.pickle - Enzymes dataset (392KB)
proteins_hypergraphs.pickle - Proteins dataset (858KB)
collab_hypergraphs.pickle - Collab dataset (9.2MB)
"""

import os
import sys
import pickle
import warnings
import argparse
from typing import Dict, Any, List, Optional


# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
# Add compute_encodings directory to path for timing_utils
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from encodings_hnns.encodings import HypergraphEncodings
from src.timing_for_encodings.timing_utils import (
    TimingCollector,
    extract_hypergraph_stats,
)

warnings.simplefilter("ignore")


def load_hypergraph_dataset(
    dataset_path: str, max_hypergraphs: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load hypergraph classification data from pickle files.

    Args:
        dataset_path: Path to the pickle file
        max_hypergraphs: Maximum number of hypergraphs to load (None for all)

    Returns:
        List of hypergraph data dictionaries

    Raises:
        FileNotFoundError: If the data file is not found
    """
    dataset_name = os.path.basename(dataset_path).replace("_hypergraphs.pickle", "")
    print(f"Loading {dataset_name} dataset from {dataset_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # Load the pickle data
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    # Handle different data formats (sometimes it's a dict, sometimes a list)
    if isinstance(data, dict):
        # Extract hypergraphs from dict structure
        hypergraphs = []
        for key, value in data.items():
            if isinstance(value, dict) and "hypergraph" in value:
                hypergraphs.append(value)
            elif isinstance(value, list):
                hypergraphs.extend(value)
    elif isinstance(data, list):
        hypergraphs = data
    else:
        raise ValueError(f"Unexpected data format in {dataset_path}")

    if max_hypergraphs is not None:
        hypergraphs = hypergraphs[:max_hypergraphs]
        print(
            f"  ✓ Loaded {len(hypergraphs)} hypergraphs (limited to {max_hypergraphs})"
        )
    else:
        print(f"  ✓ Loaded {len(hypergraphs)} hypergraphs")

    return hypergraphs


def time_single_encoding_on_hypergraph(
    hypergraph: Dict[str, Any],
    encoding_type: str,
    timing_collector: TimingCollector,
    hypergraph_index: int,
    dataset_name: str,
    **encoding_params,
) -> None:
    """Time a single encoding computation on a hypergraph.

    Args:
        hypergraph: Hypergraph dictionary
        encoding_type: Type of encoding to compute
        timing_collector: TimingCollector instance
        hypergraph_index: Index of the hypergraph being processed
        dataset_name: Name of the dataset
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
        hypergraph_copy, f"{dataset_name}_hypergraph_{hypergraph_index}"
    )
    hypergraph_stats["hypergraph_index"] = hypergraph_index
    hypergraph_stats["dataset_name"] = dataset_name

    # Initialize HypergraphEncodings
    hgencodings = HypergraphEncodings()

    # Time the specific encoding computation
    # Create dataset-specific timing key for tracking
    timing_key = f"{encoding_type}_{dataset_name}"
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


def time_encodings_on_dataset(
    hypergraphs: List[Dict[str, Any]],
    timing_collector: TimingCollector,
    encoding_types: List[str],
    dataset_name: str,
    num_hypergraphs: int = 5,
) -> None:
    """Time encoding computations on hypergraphs from a dataset.

    Args:
        hypergraphs: List of hypergraph dictionaries
        timing_collector: TimingCollector instance
        encoding_types: List of encoding types to test
        dataset_name: Name of the dataset
        num_hypergraphs: Number of hypergraphs to process
    """
    print(
        f"\n📊 Timing encoding computations on {num_hypergraphs} hypergraphs from {dataset_name}..."
    )

    for encoding_type in encoding_types:
        print(f"\n  🔍 Timing {encoding_type}...")

        successful_runs = 0
        failed_runs = 0

        for i in range(min(num_hypergraphs, len(hypergraphs))):
            print(f"    Hypergraph {i+1}/{num_hypergraphs}...", end=" ", flush=True)

            try:
                time_single_encoding_on_hypergraph(
                    hypergraph=hypergraphs[i],
                    encoding_type=encoding_type,  # Use original encoding type
                    timing_collector=timing_collector,
                    hypergraph_index=i,
                    dataset_name=dataset_name,
                )
                print("✓")
                successful_runs += 1
            except Exception as e:
                print(f"✗ Error: {str(e)[:50]}...")
                failed_runs += 1

        # Print summary for this encoding type
        if successful_runs > 0:
            print(f"    → {successful_runs}/{num_hypergraphs} runs successful")
        else:
            print(f"    → All {failed_runs} runs failed - no timing data collected")


def run_comprehensive_timing_analysis(
    data_directory: str = "data/hypergraph_classification_datasets",
    datasets: Optional[List[str]] = None,
    num_hypergraphs: Optional[int] = None,  # Default: process all hypergraphs
    encoding_types: Optional[List[str]] = None,
    max_hypergraphs_to_load: Optional[int] = None,
) -> Dict[str, TimingCollector]:
    """Run comprehensive timing analysis on hypergraph classification datasets.

    For each hypergraph individually, times:
    A) Encoding computation on hypergraphs (no lifting needed)

    Then calculates average and standard deviation per encoding type.

    Args:
        data_directory: Path to hypergraph classification datasets directory
        datasets: List of dataset names to process (None for all)
        num_hypergraphs: Number of hypergraphs to process for timing (None for all)
        encoding_types: List of encoding types to test (None for all)
        max_hypergraphs_to_load: Maximum hypergraphs to load from each dataset

    Returns:
        Dictionary mapping dataset names to TimingCollector instances
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

    if datasets is None:
        # Discover all available datasets
        available_datasets = []
        for filename in os.listdir(data_directory):
            if (
                filename.endswith("_hypergraphs.pickle")
                and "with_encodings" not in filename
            ):
                dataset_name = filename.replace("_hypergraphs.pickle", "")
                available_datasets.append(dataset_name)
        datasets = sorted(available_datasets)

    print("🚀 STARTING HYPERGRAPH CLASSIFICATION DATASETS TIMING ANALYSIS")
    print("=" * 80)
    print("Configuration:")
    print(f"  - Data directory: {data_directory}")
    print(f"  - Datasets: {', '.join(datasets)}")
    print(
        f"  - Hypergraphs to process: {'all' if num_hypergraphs is None else num_hypergraphs}"
    )
    print(f"  - Encoding types: {', '.join(encoding_types)}")

    # Store results for all datasets
    all_timing_results = {}

    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"PROCESSING DATASET: {dataset_name.upper()}")
        print(f"{'='*80}")

        try:
            # Load hypergraph data
            dataset_path = os.path.join(
                data_directory, f"{dataset_name}_hypergraphs.pickle"
            )
            hypergraphs = load_hypergraph_dataset(dataset_path, max_hypergraphs_to_load)

            # Determine actual number of hypergraphs to process
            actual_num_hypergraphs = (
                len(hypergraphs)
                if num_hypergraphs is None
                else min(num_hypergraphs, len(hypergraphs))
            )

            print(
                f"Processing {actual_num_hypergraphs} hypergraphs from {dataset_name} dataset"
            )

            # Initialize timing collector for this dataset
            timing_collector = TimingCollector()

            # Time encoding computations on hypergraphs
            time_encodings_on_dataset(
                hypergraphs,
                timing_collector,
                encoding_types,
                dataset_name,
                actual_num_hypergraphs,
            )

            all_timing_results[dataset_name] = timing_collector

        except Exception as e:
            print(f"❌ Error processing {dataset_name} dataset: {e}")
            all_timing_results[dataset_name] = TimingCollector()

    return all_timing_results


def generate_hypergraph_datasets_timing_report(
    timing_data: Dict[str, TimingCollector],
) -> str:
    """Generate a detailed timing report for hypergraph datasets analysis.

    Args:
        timing_data: Dictionary mapping dataset names to TimingCollector instances

    Returns:
        String containing detailed timing analysis report
    """
    report_lines = []
    report_lines.append("🏆 HYPERGRAPH CLASSIFICATION DATASETS TIMING ANALYSIS REPORT")
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
            # Clean encoding type name (remove dataset suffix)
            clean_encoding_type = encoding_type.split("_")[
                :-1
            ]  # Remove last part (dataset name)
            clean_encoding_type = (
                "_".join(clean_encoding_type) if clean_encoding_type else encoding_type
            )

            report_lines.append(
                f"{i:2d}. {clean_encoding_type:<20} "
                f"{stats['mean']:.4f}s ± {stats['std']:.4f}s "
                f"({stats['count']} measurements)"
            )

    # Dataset-specific analysis
    report_lines.append("\n📊 DATASET-SPECIFIC PERFORMANCE")
    report_lines.append("-" * 50)

    for dataset_name, collector in sorted(timing_data.items()):
        if not collector.timing_data:
            continue

        report_lines.append(f"\n📁 {dataset_name.upper()}:")

        dataset_stats = collector.compute_statistics()
        for encoding_type, stats in sorted(dataset_stats.items()):
            # Clean encoding type name
            clean_encoding_type = encoding_type.replace(f"_{dataset_name}", "")
            report_lines.append(
                f"   {clean_encoding_type:<20} {stats['mean']:.4f}s ± {stats['std']:.4f}s"
            )

    # Add overall summary table
    report_lines.append("\n" + combined_collector.get_timing_summary_table())

    # Add detailed report from combined collector
    report_lines.append("\n" + combined_collector.generate_timing_report())

    return "\n".join(report_lines)


def save_hypergraph_datasets_timing_results(
    timing_data: Dict[str, TimingCollector],
    output_dir: str = "computed_encodings/timing_analysis_hypergraph_datasets",
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
        print(f"💾 Saved timing data for {dataset_name}: {os.path.abspath(filename)}")

    # Combine all data and save comprehensive results
    combined_collector = TimingCollector()
    for dataset_name, collector in timing_data.items():
        for encoding_type, records in collector.timing_data.items():
            for record in records:
                combined_collector.add_timing_record(encoding_type, record)

    # Save comprehensive results
    comprehensive_file = os.path.join(
        output_dir, "hypergraph_datasets_comprehensive_timing.json"
    )
    combined_collector.save_timing_results(comprehensive_file, format_type="json")

    # Save comprehensive report
    report = generate_hypergraph_datasets_timing_report(timing_data)
    report_file = os.path.join(output_dir, "hypergraph_datasets_timing_report.txt")
    with open(report_file, "w") as f:
        f.write(report)

        print(f"💾 Saved comprehensive timing analysis: {comprehensive_file}")
    print(f"📄 Saved detailed report: {os.path.abspath(report_file)}")


def main() -> None:
    """Main function to run hypergraph datasets timing analysis."""
    parser = argparse.ArgumentParser(
        description="Timing analysis for hypergraph classification dataset encoding computations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data-directory",
        type=str,
        default="data/hypergraph_classification_datasets",
        help="Path to hypergraph classification datasets directory",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific datasets to process (default: all available)",
    )

    parser.add_argument(
        "--num-hypergraphs",
        type=int,
        default=None,
        help="Number of hypergraphs to process for timing (default: all)",
    )

    parser.add_argument(
        "--max-hypergraphs-to-load",
        type=int,
        default=None,
        help="Maximum number of hypergraphs to load from each dataset (default: None)",
    )

    parser.add_argument(
        "--encoding-types",
        nargs="+",
        default=None,
        help="Specific encoding types to test (default: all)",
    )

    parser.add_argument(
        "--lifting-method",
        type=str,
        choices=["none"],
        default="none",
        help="Lifting method - 'none' for hypergraph datasets (pre-existing hypergraphs)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="computed_encodings/timing_analysis_hypergraph_datasets",
        help="Output directory for timing results",
    )

    args = parser.parse_args()

    print("⏱️  HYPERGRAPH CLASSIFICATION DATASETS TIMING ANALYSIS")
    print("📊 PRE-EXISTING HYPERGRAPHS - ENCODING COMPUTATIONS ONLY")
    print("=" * 70)
    print("Configuration:")
    print(f"  - Data directory: {args.data_directory}")
    print(f"  - Datasets: {args.datasets or 'all available'}")
    print(
        f"  - Hypergraphs to process: {'all' if args.num_hypergraphs is None else args.num_hypergraphs}"
    )
    print(f"  - Max hypergraphs to load: {args.max_hypergraphs_to_load or 'all'}")
    print(f"  - Lifting method: {args.lifting_method} (no lifting needed)")
    print("  - Timing: Encoding computations only")
    print(f"  - Output directory: {args.output_dir}")
    if args.encoding_types:
        print(f"  - Encoding types: {', '.join(args.encoding_types)}")
    else:
        print("  - Encoding types: all (8 types)")

    try:
        # Run timing analysis
        all_timing_results = run_comprehensive_timing_analysis(
            data_directory=args.data_directory,
            datasets=args.datasets,
            num_hypergraphs=args.num_hypergraphs,
            encoding_types=args.encoding_types,
            max_hypergraphs_to_load=args.max_hypergraphs_to_load,
        )

        # Save results
        save_hypergraph_datasets_timing_results(all_timing_results, args.output_dir)

        # Print final report
        report = generate_hypergraph_datasets_timing_report(all_timing_results)
        print("\n" + report)

        print(
            f"\n🎉 Timing analysis completed! Results saved to: {os.path.abspath(args.output_dir)}"
        )

    except Exception as e:
        print(f"❌ Error during timing analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
