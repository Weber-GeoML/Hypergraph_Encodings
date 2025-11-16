#!/usr/bin/env python3
"""Timing utilities for measuring encoding computation performance.

This module provides infrastructure for collecting and analyzing timing data
for hypergraph encoding computations across different datasets.
"""

import json
import pickle
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, List

import numpy as np


class TimingCollector:
    """Collects and analyzes timing data for encoding computations.

    This class provides context managers for timing individual encoding operations
    and methods for statistical analysis of the collected timing data.
    """

    def __init__(self) -> None:
        """Initialize empty timing data collection."""
        self.timing_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    @contextmanager
    def time_encoding(self, encoding_type: str, hypergraph_stats: Dict[str, Any]):
        """Context manager for timing encoding computations.

        Args:
            encoding_type: Type of encoding being computed (e.g., 'degree', 'curvature_FRC')
            hypergraph_stats: Dictionary containing hypergraph metadata

        Yields:
            None

        Example:
            with timing_collector.time_encoding("degree", stats):
                # encoding computation here
                pass
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            computation_time = end_time - start_time

            timing_record = {
                "computation_time": computation_time,
                "start_time": start_time,
                "end_time": end_time,
                **hypergraph_stats,
            }

            self.add_timing_record(encoding_type, timing_record)

    def add_timing_record(
        self, encoding_type: str, timing_record: Dict[str, Any]
    ) -> None:
        """Add a timing record for a specific encoding type.

        Args:
            encoding_type: Type of encoding that was computed
            timing_record: Dictionary containing timing and metadata information
        """
        self.timing_data[encoding_type].append(timing_record)

    def compute_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute statistical summaries for all collected timing data.

        Returns:
            Dictionary mapping encoding types to their statistical summaries
            Format: {encoding_type: {mean: float, std: float, count: int, min: float, max: float}}
        """
        statistics_summary = {}

        for encoding_type, records in self.timing_data.items():
            if not records:
                continue

            times = [record["computation_time"] for record in records]

            stats = {
                "mean": statistics.mean(times),
                "std": statistics.stdev(times) if len(times) > 1 else 0.0,
                "count": len(times),
                "min": min(times),
                "max": max(times),
                "median": statistics.median(times),
                "times": times,  # Include raw times for further analysis
            }

            statistics_summary[encoding_type] = stats

        return statistics_summary

    def compute_dataset_statistics(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compute statistics grouped by dataset.

        Returns:
            Dictionary mapping datasets to encoding type statistics
            Format: {dataset_name: {encoding_type: {mean: float, std: float, ...}}}
        """
        dataset_stats = defaultdict(lambda: defaultdict(list))

        # Group timing data by dataset
        for encoding_type, records in self.timing_data.items():
            for record in records:
                dataset_name = record.get("dataset_name", "unknown")
                dataset_stats[dataset_name][encoding_type].append(
                    record["computation_time"]
                )

        # Compute statistics for each dataset-encoding combination
        result = {}
        for dataset_name, encoding_data in dataset_stats.items():
            result[dataset_name] = {}
            for encoding_type, times in encoding_data.items():
                if times:
                    result[dataset_name][encoding_type] = {
                        "mean": statistics.mean(times),
                        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
                        "count": len(times),
                        "min": min(times),
                        "max": max(times),
                        "median": statistics.median(times),
                    }

        return result

    def extract_hypergraph_characteristics(self) -> Dict[str, List[float]]:
        """Extract hypergraph characteristics for correlation analysis.

        Returns:
            Dictionary mapping characteristic names to lists of values
        """
        characteristics = defaultdict(list)

        for encoding_type, records in self.timing_data.items():
            for record in records:
                characteristics["num_nodes"].append(record.get("num_nodes", 0))
                characteristics["num_hyperedges"].append(
                    record.get("num_hyperedges", 0)
                )
                characteristics["mean_hyperedge_size"].append(
                    record.get("mean_hyperedge_size", 0)
                )
                characteristics["max_hyperedge_size"].append(
                    record.get("max_hyperedge_size", 0)
                )
                characteristics["computation_time"].append(record["computation_time"])

        return dict(characteristics)

    def save_timing_results(self, filepath: str, format_type: str = "json") -> None:
        """Save timing data to file.

        Args:
            filepath: Path where to save the timing data
            format_type: Format to save in ('json' or 'pickle')
        """
        data_to_save = {
            "raw_timing_data": dict(self.timing_data),
            "statistics": self.compute_statistics(),
            "dataset_statistics": self.compute_dataset_statistics(),
            "hypergraph_characteristics": self.extract_hypergraph_characteristics(),
        }

        if format_type == "json":
            with open(filepath, "w") as f:
                json.dump(data_to_save, f, indent=2, default=str)
        elif format_type == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(data_to_save, f)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def generate_timing_report(self) -> str:
        """Generate a human-readable timing report.

        Returns:
            String containing formatted timing analysis report
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("HYPERGRAPH ENCODING TIMING ANALYSIS REPORT")
        report_lines.append("=" * 80)

        # Overall statistics
        overall_stats = self.compute_statistics()
        if overall_stats:
            report_lines.append("\n📊 OVERALL TIMING STATISTICS")
            report_lines.append("-" * 50)

            for encoding_type, stats in sorted(overall_stats.items()):
                report_lines.append(f"\n🔹 {encoding_type.upper()}:")
                report_lines.append(
                    f"   Mean: {stats['mean']:.4f}s ± {stats['std']:.4f}s"
                )
                report_lines.append(
                    f"   Range: {stats['min']:.4f}s - {stats['max']:.4f}s"
                )
                report_lines.append(f"   Median: {stats['median']:.4f}s")
                report_lines.append(f"   Count: {stats['count']} computations")

        # Dataset-specific statistics
        dataset_stats = self.compute_dataset_statistics()
        if dataset_stats:
            report_lines.append("\n\n📋 DATASET-SPECIFIC TIMING STATISTICS")
            report_lines.append("-" * 50)

            for dataset_name, encoding_stats in sorted(dataset_stats.items()):
                report_lines.append(f"\n📁 Dataset: {dataset_name}")

                for encoding_type, stats in sorted(encoding_stats.items()):
                    report_lines.append(
                        f"   {encoding_type}: {stats['mean']:.4f}s ± {stats['std']:.4f}s ({stats['count']} runs)"
                    )

        # Hypergraph characteristics summary
        characteristics = self.extract_hypergraph_characteristics()
        if characteristics:
            report_lines.append("\n\n📏 HYPERGRAPH CHARACTERISTICS SUMMARY")
            report_lines.append("-" * 50)

            for char_name, values in characteristics.items():
                if values and char_name != "computation_time":
                    report_lines.append(
                        f"{char_name}: "
                        f"mean={np.mean(values):.2f}, "
                        f"std={np.std(values):.2f}, "
                        f"range=[{min(values):.0f}, {max(values):.0f}]"
                    )

        # Performance ranking
        if overall_stats:
            report_lines.append("\n\n🏆 ENCODING PERFORMANCE RANKING (by mean time)")
            report_lines.append("-" * 50)

            ranked_encodings = sorted(overall_stats.items(), key=lambda x: x[1]["mean"])
            for i, (encoding_type, stats) in enumerate(ranked_encodings, 1):
                report_lines.append(f"{i:2d}. {encoding_type}: {stats['mean']:.4f}s")

        report_lines.append("\n" + "=" * 80)

        return "\n".join(report_lines)

    def get_timing_summary_table(self) -> str:
        """Generate a concise table summary of timing data.

        Returns:
            String containing a formatted table of timing statistics
        """
        overall_stats = self.compute_statistics()
        if not overall_stats:
            return "No timing data available."

        # Create table header
        table_lines = []
        table_lines.append(
            "Encoding Type".ljust(20)
            + "Mean±Std (s)".ljust(15)
            + "Min (s)".ljust(10)
            + "Max (s)".ljust(10)
            + "Count".ljust(8)
        )
        table_lines.append("-" * 68)

        # Add data rows
        for encoding_type, stats in sorted(overall_stats.items()):
            mean_std = f"{stats['mean']:.3f}±{stats['std']:.3f}"
            row = (
                encoding_type.ljust(20)
                + mean_std.ljust(15)
                + f"{stats['min']:.3f}".ljust(10)
                + f"{stats['max']:.3f}".ljust(10)
                + str(stats["count"]).ljust(8)
            )
            table_lines.append(row)

        return "\n".join(table_lines)


def extract_hypergraph_stats(
    hypergraph: Dict[str, Any], dataset_name: str = "unknown"
) -> Dict[str, Any]:
    """Extract statistical information from a hypergraph for timing analysis.

    Args:
        hypergraph: Dictionary containing hypergraph data
        dataset_name: Name of the dataset being processed

    Returns:
        Dictionary containing hypergraph statistics
    """
    stats = {
        "dataset_name": dataset_name,
        "num_nodes": hypergraph.get("n", 0),
    }

    # Extract hypergraph structure information
    if "hypergraph" in hypergraph and isinstance(hypergraph["hypergraph"], dict):
        hg = hypergraph["hypergraph"]
        stats["num_hyperedges"] = len(hg)

        if hg:
            hyperedge_sizes = [len(edge) for edge in hg.values()]
            stats["min_hyperedge_size"] = min(hyperedge_sizes)
            stats["max_hyperedge_size"] = max(hyperedge_sizes)
            stats["mean_hyperedge_size"] = np.mean(hyperedge_sizes)
            stats["std_hyperedge_size"] = np.std(hyperedge_sizes)
        else:
            stats.update(
                {
                    "min_hyperedge_size": 0,
                    "max_hyperedge_size": 0,
                    "mean_hyperedge_size": 0,
                    "std_hyperedge_size": 0,
                }
            )

    # Extract feature information
    if "features" in hypergraph:
        features = hypergraph["features"]
        if hasattr(features, "shape"):
            stats["num_features"] = features.shape[1] if len(features.shape) > 1 else 1
        else:
            stats["num_features"] = 0

    return stats
