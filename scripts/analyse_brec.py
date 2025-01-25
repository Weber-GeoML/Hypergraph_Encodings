"""
This script is used to analyse the BREC dataset.
It is used to compare the encodings of the graphs in the BREC dataset.
"""

import networkx as nx
from brec.dataset import BRECDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from encodings_hnns.liftings_and_expansions import lift_to_hypergraph
from brec_analysis.utils_for_brec import (
    create_output_dirs,
    convert_nx_to_hypergraph_dict,
    nx_to_pyg,
)
from brec_analysis.plotting_graphs_and_hgraphs_for_brec import (
    plot_graph_pair,
    plot_hypergraph_pair,
)
from brec_analysis.compare_encodings_wrapper import compare_encodings_wrapper
from brec_analysis.match_status import MatchStatus
import json
from brec_analysis.analyse_brec_categories import analyze_brec_categories
from brec_analysis.southern_orc_example import southern_orc_example
import click
from brec_analysis.encodings_to_check import ENCODINGS_TO_CHECK
import multiprocessing as mp

# Define valid categories with their ranges
VALID_CATEGORIES = [
    ("basic", (0, 60), "Basic"),
    ("regular", (60, 85), "Regular"),
    ("str", (85, 135), "Strongly Regular"),
    ("cfi", (135, 185), "CFI"),
    ("extension", (185, 235), "Extension"),
    ("4vtx", (235, 255), "4-Vertex Condition"),
    ("dr", (255, 275), "Distance Regular"),
]

PART_DICT = {
    "Basic": (0, 60),
    "Regular": (60, 85),
    "strongly regular": (85, 135),
    "CFI": (135, 185),
    "Extension": (185, 235),
    "4-Vertex_Condition": (235, 255),
    "Distance_Regular": (255, 275),
}


def analyze_graph_pair(
    data1: Data,
    data2: Data,
    pair_idx: int | str,
    category: str,
    is_isomorphic: bool,
    already_in_nx: bool = False,
    types_of_encoding: list[tuple[str, str]] = ENCODINGS_TO_CHECK,
) -> dict:
    """Analyze a pair of graphs: plot them and compare their encodings

    Args:
        data1 (Data):
            The first graph.
        data2 (Data):
            The second graph.
        pair_idx (str):
            The index of the pair.
        category (str):
            The category of the pair.
        is_isomorphic (bool):
            Whether the graphs are isomorphic.
        encoding (str):
            The encoding to use.
    """
    if not already_in_nx:
        # Convert PyG data to NetworkX graphs
        G1 = to_networkx(data1, to_undirected=True)
        G2 = to_networkx(data2, to_undirected=True)
    else:
        G1 = data1
        G2 = data2

    assert not is_isomorphic, "All pairs in BREC are non-isomorphic"

    # store the Asjacency matrix plots and their difference

    # Store the node mapping if graphs are isomorphic
    node_mapping = None
    if is_isomorphic:
        node_mapping = find_isomorphism_mapping(G1, G2)
        if node_mapping is None:
            print(
                f"WARNING: Pair {pair_idx} is marked as isomorphic but no isomorphism found!"
            )
        else:
            print(f"\nIsomorphism mapping for pair {pair_idx}:")
            print("G1 node -> G2 node")
            for node1, node2 in node_mapping.items():
                print(f"{node1} -> {node2}")

    # Plot original graphs
    # in graph space
    plot_graph_pair(G1, G2, pair_idx, category, is_isomorphic, "plots/graph_pairs")

    # Convert to hypergraph dictionaries
    # THESE ARE STILL GRAPHS!!!
    hg1 = convert_nx_to_hypergraph_dict(G1)
    hg2 = convert_nx_to_hypergraph_dict(G2)

    """
    'hypergraph': {'e_0': [...], 'e_1': [...],
    'features': tensor([], size=(10, 0)), 'labels': {}, 'n': 10}
    """
    # Initialize the results structure
    final_results: dict = {
        "pair_idx": pair_idx,
        "category": category,
        "is_isomorphic": is_isomorphic,
        "graph_level": {"encodings": {}},
        "hypergraph_level": {"encodings": {}},
    }

    # Compare graph-level encodings
    print(f"\nAnalyzing pair {pair_idx} ({category}):")
    print("\n")
    graph_results = compare_encodings_wrapper(
        hg1,
        hg2,
        pair_idx,
        category,
        is_isomorphic,
        "graph",
        node_mapping,
        types_of_encoding=types_of_encoding,
    )
    final_results["graph_level"]["encodings"] = graph_results["encodings"]

    del hg1, hg2

    print("*-" * 25)
    print("*-" * 25)
    print(f"Analyzing pair {pair_idx} ({category}): at the hypergraph level")
    print("*-" * 25)
    print("*-" * 25)

    # Lift to hypergraphs and compare
    hg1_lifted = lift_to_hypergraph(data1, verbose=False, already_in_nx=already_in_nx)
    hg2_lifted = lift_to_hypergraph(data2, verbose=False, already_in_nx=already_in_nx)

    plot_hypergraph_pair(
        G1,
        G2,
        hg1_lifted,
        hg2_lifted,
        pair_idx,
        category,
        is_isomorphic,
        "plots/hypergraph_pairs",
    )

    # Compare hypergraph-level encodings
    hypergraph_results = compare_encodings_wrapper(
        hg1_lifted,
        hg2_lifted,
        pair_idx,
        category,
        is_isomorphic,
        "hypergraph",
        node_mapping,
        types_of_encoding=types_of_encoding,
    )
    final_results["hypergraph_level"]["encodings"] = hypergraph_results["encodings"]

    return final_results


def write_results(f, results: dict, json_results: dict) -> None:
    """Write results for a pair to the file and update JSON results."""
    # Print the file path at the start
    print(f"\nWriting results to: {f.name}")

    # Get pair info
    pair_idx = results["pair_idx"]
    category = results["category"]

    # Initialize this pair in json_results if needed
    if category not in json_results:
        json_results[category] = {}

    for level in ["graph_level", "hypergraph_level"]:
        f.write(f"\nAnalysis for pair {pair_idx} ({category}) - {level}\n")

        for encoding_type, result in results[level]["encodings"].items():
            # Write to text file
            f.write(f"\n=== {result[level]['encodings']} ===\n")
            is_same = result["status"] in [
                MatchStatus.EXACT_MATCH,
                MatchStatus.SCALED_MATCH,
            ]
            f.write(f"Result: {'Same' if is_same else 'Different'}\n")
            if result["status"] == MatchStatus.SCALED_MATCH:
                f.write(f"Scaling factor: {result['scaling_factor']}\n")

            # Update JSON structure
            encoding_key = f"{level.split('_')[0].capitalize()} ({encoding_type})"
            if encoding_key not in json_results[category]:
                json_results[category][encoding_key] = {"different": 0, "total": 0}

            json_results[category][encoding_key]["total"] += 1
            if not is_same:
                json_results[category][encoding_key]["different"] += 1


def quick_eda_from_github(graphs):
    # Now you can analyze specific categories or pairs
    # For example, analyze the first pair of basic graphs:
    if "basic" in graphs:
        G1, G2 = graphs["basic"][0], graphs["basic"][1]
        print("\nAnalyzing first pair of basic graphs:")
        print(f"G1: {G1.number_of_nodes()} nodes, {G1.number_of_edges()} edges")
        print(f"G2: {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")
    if "regular" in graphs:
        G1, G2 = graphs["regular"][0], graphs["regular"][1]
        print("\nAnalyzing first pair of regular graphs:")
        print(f"G1: {G1.number_of_nodes()} nodes, {G1.number_of_edges()} edges")
        print(f"G2: {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")
        # degree distribution
        print(f"G1 degree distribution: {G1.degree()}")
        print(f"G2 degree distribution: {G2.degree()}")


def rook_and_shrikhande_special_case() -> None:
    """Analyze the Rook and Shrikhande graphs"""
    # Create results files
    results_file = "results/brec/rook_vs_shrikhande_comparisons.txt"
    json_file = "results/brec/rook_vs_shrikhande_statistics.json"

    # Initialize JSON results
    json_results: dict = {}
    # First analyze Rook and Shrikhande graphs
    print("\nAnalyzing Rook and Shrikhande graphs...")

    # Load the graphs
    rook = nx.read_graph6("rook_graph.g6")
    shrikhande = nx.read_graph6("shrikhande.g6")

    rook_data = nx_to_pyg(rook)
    shrikhande_data = nx_to_pyg(shrikhande)

    # Analyze as a special pair
    print("Analyzing Rook vs Shrikhande")

    # After loading rook and shrikhande graphs
    print("Computing ORCs for Rook and Shrikhande graphs...")

    print("Using the Southern ORC example")
    southern_orc_example(rook, shrikhande)
    print("Done with Southern ORC example")

    special_results = analyze_graph_pair(
        rook_data,
        shrikhande_data,
        pair_idx="rook_vs_shrikhande",
        category="Special",
        is_isomorphic=False,
    )
    with open(results_file, "w") as f:
        write_results(f, special_results, json_results)


def parse_encoding(encodings: str) -> list[tuple[str, str]]:
    """Parse encoding indices"""
    # Parse encoding indices
    try:
        if "-" in encodings:
            # Handle range format (e.g., "0-3")
            start, end = map(int, encodings.split("-"))
            encoding_indices = list(range(start, end + 1))
            print(f"start: {start}, end: {end}, encoding_indices: {encoding_indices}")
        else:
            # Handle comma-separated format (e.g., "0,2,3")
            encoding_indices = [int(i) for i in encodings.split(",")]

        # Validate indices
        max_idx = len(ENCODINGS_TO_CHECK) - 1
        invalid_indices = [i for i in encoding_indices if i < 0 or i > max_idx]
        if invalid_indices:
            raise ValueError(
                f"Invalid encoding indices: {invalid_indices}. "
                f"Must be between 0 and {max_idx}"
            )

        # Select specified encodings
        selected_encodings = [ENCODINGS_TO_CHECK[i] for i in encoding_indices]
        return selected_encodings
    except ValueError as e:
        print(f"Error parsing encoding indices: {e}")
        print("\nAvailable encodings:")
        for i, (code, name) in enumerate(ENCODINGS_TO_CHECK):
            print(f"{i}: {code} ({name})")
        return


def parse_categories(categories_str: str) -> list[str]:
    """Parse category indices from string input.

    Args:
        categories_str: String containing category indices (e.g., "0" or "0,3" or "0-3")

    Returns:
        List of selected category names

    Raises:
        ValueError: If invalid indices are provided
    """
    try:
        # Handle range format (e.g., "0-3")
        if "-" in categories_str:
            start, end = map(int, categories_str.split("-"))
            category_indices = list(range(start, end + 1))
        else:
            # Handle comma-separated format (e.g., "0,2,3")
            category_indices = [int(i) for i in categories_str.split(",")]

        # Validate indices
        max_idx = len(VALID_CATEGORIES) - 1
        invalid_indices = [i for i in category_indices if i < 0 or i > max_idx]
        if invalid_indices:
            raise ValueError(
                f"Invalid category indices: {invalid_indices}\n"
                f"Must be between 0 and {max_idx}"
            )

        # Get selected categories
        selected_categories = [VALID_CATEGORIES[i][0] for i in category_indices]

        # Print selection info
        print("\nSelected categories:")
        for idx in category_indices:
            code, range_, name = VALID_CATEGORIES[idx]
            print(f"  {name} (range: {range_[0]}-{range_[1]})")

        return selected_categories

    except ValueError as e:
        print(f"Error parsing category indices: {e}")
        print("\nAvailable categories:")
        for i, (code, range_, name) in enumerate(VALID_CATEGORIES):
            print(f"{i}: {name} (range: {range_[0]}-{range_[1]})")
        raise


@click.command()
@click.option(
    "--encodings",
    "-e",
    help='Indices of encodings to check (e.g., "0" or "0,3" or "0-3")',
    default="0",
)
@click.option(
    "--categories",
    "-c",
    help='Indices of categories to analyze (e.g., "0" or "0,3" or "0-3")',
    default="0",
)
def main(encodings: str, categories: str) -> None:
    """Analyze BREC dataset with specified encodings and categories

    Args:
        encodings: Comma-separated indices or range (e.g., "0,2,3" or "0-3")
        categories: Comma-separated categories to analyze (e.g., "basic,regular" or "all")
    """

    selected_encodings = parse_encoding(encodings)
    selected_categories = parse_categories(categories)

    print(f"Selected encodings: {selected_encodings}")
    print(f"Selected categories: {selected_categories}")

    create_output_dirs()

    rook_and_shrikhande_special_case()

    dataset_pip = BRECDataset()
    print(f"Total number of graphs in BREC dataset: {len(dataset_pip)}")

    # Get the graphs for analysis
    graphs_read_from_files: dict[str, list[nx.Graph]] = analyze_brec_categories()
    quick_eda_from_github(graphs_read_from_files)

    # BREC with pip package
    # help(BRECDataset)
    # or
    # from brec import __version__
    # print(f"BREC version: {__version__}")

    # print(f"Total number of graphs in BREC dataset: {len(dataset)}")

    # # After loading dataset
    # print("\nDataset Summary:")
    # dataset_pip.print_summary()

    # # We can also try to get the number of classes
    # print(f"\nNumber of classes: {dataset_pip.num_classes}")

    # # Since BREC contains pairs of graphs, let's show the breakdown
    # num_pairs = len(dataset_pip) // 2
    # print(f"Number of graph pairs: {num_pairs}")
    # # After loading dataset
    # first_graph = dataset_pip[0]
    # print("First graph object:")
    # print(first_graph)
    # print("\nObject type:", type(first_graph))
    # print("\nAvailable attributes:")
    # for attr in dir(first_graph):
    #     if not attr.startswith('_'):  # Skip private attributes
    #         print(f"{attr}: {getattr(first_graph, attr)}")
    # assert False

    # Create results files
    results_file = "results/brec/all_comparisons.txt"
    json_file = "results/brec/statistics.json"

    # Initialize JSON results
    json_results: dict = {}

    print("*-" * 25)
    print("*-" * 25)
    print("*-" * 25)
    print("*-" * 25)
    print("*-" * 25)
    print("*-" * 25)
    print("Systematically analyzing the BREC dataset")

    # Create results directory first
    create_output_dirs()

    with open(results_file, "w") as f:

        # Then continue with BREC dataset analysis
        # After loading the dataset
        # Show breakdown by category

        print("\nBreakdown by category:")
        for category, (start, end) in PART_DICT.items():
            num_category_pairs = end - start
            print(
                f"{category}: {num_category_pairs} pairs ({num_category_pairs*2} graphs)"
            )

        pip = False
        if pip:
            # Process pairs and collect results
            for category, (start, end) in PART_DICT.items():
                print(f"\nProcessing {category} category...")
                for pair_idx in range(start, end):
                    pair_results = analyze_graph_pair(
                        dataset[pair_idx * 2],
                        dataset[pair_idx * 2 + 1],
                        pair_idx,
                        category,
                        is_isomorphic=False,
                        types_of_encoding=selected_encodings,
                    )
                    write_results(f, pair_results, json_results)

        else:
            print(f"Only checking these types of encodings: {selected_encodings}")
            print(f"Only checking these categories: {selected_categories}")
            total_pair_idx = 0  # Keep track of total pairs processed

            for category, graphs in graphs_read_from_files.items():
                if category not in selected_categories:
                    print(f"Skipping {category} category")
                    continue

                print(f"\nProcessing {category} category...")
                num_pairs_to_process = min(5, len(graphs) // 2)

                for local_pair_idx in range(num_pairs_to_process):
                    g1 = graphs[local_pair_idx * 2]
                    g2 = graphs[local_pair_idx * 2 + 1]

                    # Skip if either graph has more than 20 nodes
                    if g1.number_of_nodes() > 20 or g2.number_of_nodes() > 20:
                        print(
                            f"Skipping pair {local_pair_idx + 1}/5 in {category} "
                            f"(too many nodes: G1={g1.number_of_nodes()}, "
                            f"G2={g2.number_of_nodes()})"
                        )
                        continue

                    print(
                        f"Processing pair {local_pair_idx + 1}/5 in {category} "
                        f"(global pair index: {total_pair_idx})"
                    )

                    pair_results = analyze_graph_pair(
                        g1,
                        g2,
                        total_pair_idx,  # Use the global pair index
                        category,
                        is_isomorphic=False,
                        already_in_nx=True,
                        types_of_encoding=selected_encodings,
                    )
                    write_results(f, pair_results, json_results)

                    total_pair_idx += 1  # Increment the global counter

        # Save JSON results with percentages
        final_stats: dict = {}
        for category in json_results:
            final_stats[category] = {}
            for enc in json_results[category]:
                stats = json_results[category][enc]
                percentage = (
                    (stats["different"] / stats["total"]) * 100
                    if stats["total"] > 0
                    else 0
                )
                final_stats[category][enc] = round(percentage, 2)

        # Save JSON file
        with open(json_file, "w") as json_f:
            json.dump(final_stats, json_f, indent=2)

        print(f"\nStatistics saved to: {json_file}")

        # Generate LaTeX table
        generate_latex_table(final_stats)

    # # Check each pair in each category
    # for category, (start, end) in PART_DICT.items():
    #     print(f"\nChecking connectivity for {category} category...")
    #     for pair_idx in range(start, end):
    #         # Get both graphs in the pair
    #         G1 = to_networkx(dataset[pair_idx * 2], to_undirected=True)
    #         G2 = to_networkx(dataset[pair_idx * 2 + 1], to_undirected=True)

    #         # Check connectivity
    #         is_G1_connected = nx.is_connected(G1)
    #         is_G2_connected = nx.is_connected(G2)

    #         connectivity_stats[category]["total_pairs"] += 1

    #         # If both graphs are connected
    #         if is_G1_connected and is_G2_connected:
    #             connectivity_stats[category]["connected"] += 1
    #         else:
    #             connectivity_stats[category]["disconnected"] += 1
    #             # Print which graphs are disconnected and plot them
    #             if not is_G1_connected and not is_G2_connected:
    #                 print(f"  Pair {pair_idx}: Both graphs are disconnected")
    #             elif not is_G1_connected:
    #                 print(f"  Pair {pair_idx}: First graph is disconnected")
    #             else:
    #                 print(f"  Pair {pair_idx}: Second graph is disconnected")

    #             # Plot the disconnected pair
    #             plot_disconnected_pair(G1, G2, pair_idx, category)
    #             print(
    #                 f"  Plot saved to: plots/disconnected_pairs/pair_{pair_idx}_{category.lower()}.png"
    #             )

    # # Print summary
    # print("\nConnectivity Summary:")
    # print("-" * 60)
    # print(f"{'Category':<20} {'Connected':<12} {'Disconnected':<12} {'Total':<12}")
    # print("-" * 60)

    # for category in connectivity_stats:
    #     stats = connectivity_stats[category]
    #     connected_percent = (stats["connected"] / stats["total_pairs"]) * 100
    #     print(
    #         f"{category:<20} {stats['connected']:<12} {stats['disconnected']:<12} {stats['total_pairs']:<12} ({connected_percent:.1f}% connected)"
    #     )

    # return connectivity_stats


def process_pair(dataset: BRECDataset, encoding: str, pair_info: tuple) -> dict:
    """Process a single pair of graphs."""
    category, pair_idx = pair_info
    print(f"\nDEBUG: Processing pair_info: {pair_info}")
    print(f"DEBUG: Category: {category}, Pair Index: {pair_idx}")
    print(f"DEBUG: Dataset indices: {pair_idx * 2} and {pair_idx * 2 + 1}")

    # Get both graphs and check connectivity
    G1 = to_networkx(dataset[pair_idx * 2], to_undirected=True)
    G2 = to_networkx(dataset[pair_idx * 2 + 1], to_undirected=True)

    # Check regularity for Regular category
    if category == "Regular":
        is_reg1, deg1 = is_regular(G1)
        is_reg2, deg2 = is_regular(G2)
        assert (
            is_reg1
        ), f"Graph 1 in Regular pair {pair_idx} is not regular! Degrees: {[d for _, d in G1.degree()]}"
        assert (
            is_reg2
        ), f"Graph 2 in Regular pair {pair_idx} is not regular! Degrees: {[d for _, d in G2.degree()]}"
        print(f"Regular graphs confirmed: degrees {deg1} and {deg2}")

    # Skip if either graph is disconnected
    if not (nx.is_connected(G1) and nx.is_connected(G2)):
        print(f"Skipping pair {pair_idx} ({category}): Contains disconnected graph(s)")
        return None

    # Analyze connected pair
    pair_results = analyze_graph_pair(
        dataset[pair_idx * 2],
        dataset[pair_idx * 2 + 1],
        pair_idx,
        category,
        is_isomorphic=False,
        encoding=encoding,
    )

    return pair_results

    # print(f"\nAnalyzing BREC dataset with {encoding} encoding...")
    # print(f"Using {num_workers} worker processes")

    # create_output_dirs()
    # dataset: BRECDataset = BRECDataset()

    # # First check connectivity
    # print("\nAnalyzing connectivity of BREC dataset...")
    # connectivity_stats: dict = check_connectivity_stats(dataset)

    # # Save connectivity stats to JSON
    # connectivity_file: str = "results/connectivity_stats.json"
    # with open(connectivity_file, "w") as f:
    #     json.dump(connectivity_stats, f, indent=2)
    # print(f"\nConnectivity statistics saved to: {connectivity_file}")


def generate_latex_table(stats: dict) -> None:
    """Generate LaTeX table from statistics."""
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

    latex_file: str = "results/comparison_table.tex"
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


def is_regular(G: nx.Graph) -> tuple[bool, int]:
    """Check if a graph is regular (all nodes have same degree).

    Args:
        G: NetworkX graph

    Returns:
        tuple[bool, int]: (is_regular, degree if regular else -1)
    """
    if len(G) == 0:  # Empty graph
        return True, 0

    degrees = [d for _, d in G.degree()]
    first_degree = degrees[0]

    # Check if all degrees are equal to the first degree
    is_reg = all(d == first_degree for d in degrees)

    return is_reg, first_degree if is_reg else -1


if __name__ == "__main__":
    # Required for Windows compatibility
    mp.freeze_support()
    main()
