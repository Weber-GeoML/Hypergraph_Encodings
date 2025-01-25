import glob
import json

from brec_analysis.statistics import generate_latex_table


def combine_results():
    """Combine results from all encoding analyses."""
    # Get all statistics files
    stat_files = glob.glob("results/statistics_*.json")

    # Combined results
    combined_stats = {}

    # Read each file
    for stat_file in stat_files:
        with open(stat_file) as f:
            stats = json.load(f)
            for category in stats:
                if category not in combined_stats:
                    combined_stats[category] = {}
                combined_stats[category].update(stats[category])

    # Save combined results
    with open("results/combined_statistics_not_sure.json", "w") as f:
        json.dump(combined_stats, f, indent=2)

    # Generate final LaTeX table
    generate_latex_table(
        combined_stats, latex_file="results/comparison_table_not_sure.tex"
    )


if __name__ == "__main__":
    combine_results()
