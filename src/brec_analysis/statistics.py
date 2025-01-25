"""Functions for calculating statistics and generating LaTeX tables."""

from typing import Dict

from brec_analysis.match_status import MatchStatus


def calculate_statistics(
    results_by_category: dict, consider_scaling: bool = True
) -> Dict[str, Dict[str, float]]:
    """Calculate percentage of different encodings for each category.

    Args:
        results_by_category: Dictionary with results by category
        consider_scaling: If True, count SCALED_MATCH as a match

    Returns:
        Dictionary with statistics by category and encoding
    """
    stats = {}
    for category in results_by_category:
        stats[category] = {}
        for level in ["graph_level", "hypergraph_level"]:
            for encoding in results_by_category[category][level]:
                results = results_by_category[category][level][encoding]

                if consider_scaling:
                    # Count as different only if NO_MATCH
                    different = sum(1 for r in results if r == MatchStatus.NO_MATCH)
                else:
                    # Count as different if not EXACT_MATCH
                    different = sum(1 for r in results if r != MatchStatus.EXACT_MATCH)

                total = len(results)
                if total > 0:  # Avoid division by zero
                    percentage = (different / total) * 100
                    key = f"{level}_{encoding}"
                    stats[category][key] = percentage

    return stats


def generate_latex_table(
    stats: dict, latex_file: str = "results/comparison_table.tex"
) -> None:
    """Generate LaTeX table from statistics.

    Args:
        stats:
            Dictionary with statistics
        latex_file:
            File to save the LaTeX table
    """
    categories: list[str] = [
        "Basic",
        "Regular",
        "Extension",
        "CFI",
        "4-Vertex_Condition",
    ]
    encodings: list[str] = [
        "Graph (1-WL)",
        "Hypergraph (1-WL)",
        "Graph (LDP)",
        "Hypergraph (LDP)",
        "Graph (LCP-FRC)",
        "Hypergraph (LCP-FRC)",
        "Graph (EE RWPE)",
        "Graph (EN RWPE)",
        "Graph (Hodge LAPE)",
        "Graph (Normalized LAPE)",
    ]

    with open(latex_file, "w") as f:
        f.write("\\begin{table*}[h!]\n\\centering\n\\tiny\n")
        f.write("\\begin{tabular}{|l|" + "c|" * len(categories) + "}\n\\hline\n")

        # Header
        f.write(
            "\\textbf{Level (Encodings)} & "
            + " & ".join([f"\\textbf{{{cat}}}" for cat in categories])
            + " \\\\\n\\hline\n"
        )

        # Data rows
        for encoding in encodings:
            row = [encoding]
            for category in categories:
                value = stats.get(category, {}).get(encoding, "")
                row.append(f"{value}\\%" if value != "" else "")
            f.write(" & ".join(row) + " \\\\\n")

        # Table footer
        f.write("\\hline\n\\end{tabular}\n")
        f.write(
            "\\caption{Difference in encodings on BREC dataset. We report the percentage of pairs with different encoding, at different level (graph or hypergraph)}\n"
        )
        f.write("\\end{table*}\n")

    print(f"\nLaTeX table saved to: {latex_file}")
