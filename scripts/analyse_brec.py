"""
This script is used to analyse the BREC dataset.
It is used to compare the encodings of the graphs in the BREC dataset.
"""

import json
import multiprocessing as mp
import os

import click
import networkx as nx
from brec.dataset import BRECDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from brec_analysis.analyse_brec_categories import (
    analyze_brec_categories,
    quick_eda_from_github,
)
from brec_analysis.categories_to_check import PART_DICT
from brec_analysis.compare_encodings_wrapper import compare_encodings_wrapper
from brec_analysis.encodings_to_check import ENCODINGS_TO_CHECK
from brec_analysis.isomorphism_mapping import find_isomorphism_mapping
from brec_analysis.match_status import MatchStatus
from brec_analysis.parse_click_args import parse_categories, parse_encoding
from brec_analysis.plotting_graphs_and_hgraphs_for_brec import (
    plot_graph_pair,
    plot_hypergraph_pair,
)
from brec_analysis.southern_orc_example import southern_orc_example
from brec_analysis.statistics import generate_latex_table
from brec_analysis.utils_for_brec import (
    convert_nx_to_hypergraph_dict,
    create_output_dirs,
    nx_to_pyg,
)
from encodings_hnns.liftings_and_expansions import lift_to_hypergraph


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


def write_results(f, filepath_json, results: dict, json_results: dict) -> dict:
    """Write results for a pair to the file and update JSON results.

    Args:
        f:
            the file to write to
        filepath_json:
            the path to the json file
        results:
            the results to write
        json_results:
            the json results to update


    Returns:
        the updated json results

    results = {'pair_idx': 'rook_vs_shrikhande', 'category': 'Special', 'is_isomorphic': False, 'graph_level': {'encodings': {...}}, 'hypergraph_level': {'encodings': {...}}}

    results["graph_level"]["encodings"]
    {'LDP': {'description': 'Local Degree Profile', 'status': <MatchStatus.EXACT_MATCH: 'EXACT_MATCH'>, 'scaling_factor': 1.0, 'permutation': (...)}, 'LCP-FRC': {'description': 'Local Curvature Profile - FRC', 'status': <MatchStatus.EXACT_MATCH: 'EXACT_MATCH'>, 'scaling_factor': 1.0, 'permutation': (...)}, 'RWPE': {'description': 'Random Walk Encodings', 'status': <MatchStatus.EXACT_MATCH: 'EXACT_MATCH'>, 'scaling_factor': 1.0, 'permutation': (...)}, 'LCP-ORC': {'description': 'Local Curvature Profile - ORC', 'status': <MatchStatus.SCALED_MATCH: 'SCALED_MATCH'>, 'scaling_factor': 0.4954381921284782, 'permutation': (...)}, 'LAPE-Normalized': {'description': 'Normalized Laplacian', 'status': <MatchStatus.NO_MATCH: 'NO_MATCH'>, 'scaling_factor': None, 'permutation': None}, 'LAPE-RW': {'description': 'Random Walk Laplacian', 'status': <MatchStatus.NO_MATCH: 'NO_MATCH'>, 'scaling_factor': None, 'permutation': None}, 'LAPE-Hodge': {'description': 'Hodge Laplacian', 'status': <MatchStatus.NO_MATCH: 'NO_MATCH'>, 'scaling_factor': None, 'permutation': None}}
    """
    # Print the file path at the start
    print(f"\nWriting results to: {f.name}")
    print(f"Results: {results}")

    # Get pair info
    pair_idx = results["pair_idx"]
    category = results["category"]

    # Initialize this pair in json_results if needed
    if category not in json_results:
        json_results[category] = {}

    # Initialize simple JSON format
    simple_result = {
        "pair": pair_idx,
        "category": category,
        "encodings": {"graph": {}, "hypergraph": {}},
    }

    for level in ["graph_level", "hypergraph_level"]:
        f.write(f"\nAnalysis for pair {pair_idx} ({category}) - {level}\n")
        level_key = "graph" if level == "graph_level" else "hypergraph"

        for encoding_type, encoding_results in results[level]["encodings"].items():
            # Write to text file
            f.write(f"\n=== {encoding_type} ({encoding_results['description']}) ===\n")

            status = encoding_results["status"]
            result = {
                MatchStatus.EXACT_MATCH: "Same",
                MatchStatus.SCALED_MATCH: "Scaled",
                MatchStatus.NO_MATCH: "Different",
                MatchStatus.TIMEOUT: "Timeout",
            }.get(
                status, "Different"
            )  # Default to "Different" for unknown status
            f.write(f"Result: {result}\n")
            is_same = result == "Same"
            if encoding_results["status"] == MatchStatus.SCALED_MATCH:
                f.write(f"Scaling factor: {encoding_results['scaling_factor']}\n")

            # Update complex JSON structure
            encoding_key = f"{level.split('_')[0].capitalize()} ({encoding_type})"
            if encoding_key not in json_results[category]:
                json_results[category][encoding_key] = {
                    "different": 0,
                    "same": 0,
                    "same_with_timeout": 0,
                    "total": 0,
                    "total_with_timeout": 0,
                }

            if not result == "Timeout":
                json_results[category][encoding_key]["total"] += 1
                if not is_same:
                    json_results[category][encoding_key]["different"] += 1
                else:
                    json_results[category][encoding_key]["same"] += 1
            else:
                json_results[category][encoding_key]["same_with_timeout"] += 1
            json_results[category][encoding_key]["total_with_timeout"] += 1

            # Update simple JSON structure
            if is_same:
                result_to_write = "Same"
            elif result == "Timeout":
                result_to_write = result
            else:
                result_to_write = "Different"
            simple_result["encodings"][level_key][encoding_type] = {
                "result": result_to_write,
                "scaling_factor": (
                    encoding_results["scaling_factor"]
                    if encoding_results["status"] == MatchStatus.SCALED_MATCH
                    else None
                ),
            }

    # save the json
    with open(filepath_json, "w") as json_f:
        json.dump(simple_result, json_f, indent=2)

    return simple_result


def rook_and_shrikhande_special_case() -> None:
    """Analyze the Rook and Shrikhande graphs"""
    # Create results files
    results_file = "results/brec/rook_vs_shrikhande_comparisons.txt"
    json_path = "results/brec/rook_vs_shrikhande_statistics.json"

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
        write_results(f, json_path, special_results, json_results)

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


# def process_pair(dataset: BRECDataset, encoding: str, pair_info: tuple) -> dict | None:
#     """Process a single pair of graphs."""
#     category, pair_idx = pair_info
#     print(f"\nDEBUG: Processing pair_info: {pair_info}")
#     print(f"DEBUG: Category: {category}, Pair Index: {pair_idx}")
#     print(f"DEBUG: Dataset indices: {pair_idx * 2} and {pair_idx * 2 + 1}")

#     # Get both graphs and check connectivity
#     G1 = to_networkx(dataset[pair_idx * 2], to_undirected=True)
#     G2 = to_networkx(dataset[pair_idx * 2 + 1], to_undirected=True)

#     # Check regularity for Regular category
#     if category == "Regular":
#         is_reg1, deg1 = is_regular(G1)
#         is_reg2, deg2 = is_regular(G2)
#         assert (
#             is_reg1
#         ), f"Graph 1 in Regular pair {pair_idx} is not regular! Degrees: {[d for _, d in G1.degree()]}"
#         assert (
#             is_reg2
#         ), f"Graph 2 in Regular pair {pair_idx} is not regular! Degrees: {[d for _, d in G2.degree()]}"
#         print(f"Regular graphs confirmed: degrees {deg1} and {deg2}")

#     # Skip if either graph is disconnected
#     if not (nx.is_connected(G1) and nx.is_connected(G2)):
#         print(f"Skipping pair {pair_idx} ({category}): Contains disconnected graph(s)")
#         return None

#     # Analyze connected pair
#     pair_results = analyze_graph_pair(
#         dataset[pair_idx * 2],
#         dataset[pair_idx * 2 + 1],
#         pair_idx,
#         category,
#         is_isomorphic=False,
#         types_of_encoding=list(encoding),
#     )

#     return pair_results

# print(f"\nAnalyzing BREC dataset with {encoding} encoding...")
# print(f"Using {num_workers} worker processes")

# create_output_dirs()
# dataset: BRECDataset = BRECDataset()

# # First check connectivity
# print("\nAnalyzing connectivity of BREC dataset...")
# connectivity_stats: dict = check_connectivity_stats(dataset)

# # Save connectivity stats to JSON
# connectivity_file: str = "results/brec/connectivity_stats.json"
# with open(connectivity_file, "w") as f:
#     json.dump(connectivity_stats, f, indent=2)
# print(f"\nConnectivity statistics saved to: {connectivity_file}")


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
        encodings:
            Comma-separated indices or range (e.g., "0,2,3" or "0-3")
            Dictates which encodings to check
        categories:
            Comma-separated categories to analyze (e.g., "basic,regular" or "all")
            Dictates which BREC categories to analyze
    """

    selected_encodings = parse_encoding(encodings)
    selected_categories = parse_categories(categories)

    print(f"Selected encodings: {selected_encodings}")
    print(f"Selected categories: {selected_categories}")

    create_output_dirs()

    # rook_and_shrikhande_special_case()

    # dataset_pip = BRECDataset()
    # print(f"Total number of graphs in BREC dataset: {len(dataset_pip)}")

    # Get the graphs for analysis
    graphs_read_from_files: dict[str, list[nx.Graph]] = analyze_brec_categories()
    quick_eda_from_github(graphs_read_from_files, verbose=False)
    # loop through all graphs_read_from_files, assert the grabs are connected
    for category, graphs in graphs_read_from_files.items():
        for graph in graphs:
            assert nx.is_connected(graph), f"Graph in {category} is not connected"
            # assert the number of connected components is 1
            components = list(nx.connected_components(graph))
            assert (
                len(components) == 1
            ), f"Graph in {category} has {len(components)} components"

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
    results_file = f"results/brec/all_comparisons_encodings_{encodings}_categories_{categories}.txt"
    json_file = (
        f"results/brec/statistics_encodings_{encodings}_categories_{categories}.json"
    )

    print(f"results_file: {results_file}")
    print(f"json_file: {json_file}")

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

        print(f"Only checking these types of encodings: {selected_encodings}")
        print(f"Only checking these categories: {selected_categories}")

        pip = False
        # if pip:
        #     # Process pairs and collect results
        #     for category, (start, end) in PART_DICT.items():
        #         print(f"\nProcessing {category} category...")
        #         for pair_idx in range(start, end):
        #             pair_results = analyze_graph_pair(
        #                 dataset[pair_idx * 2],
        #                 dataset[pair_idx * 2 + 1],
        #                 pair_idx,
        #                 category,
        #                 is_isomorphic=False,
        #                 types_of_encoding=selected_encodings,
        #             )
        #             if len(selected_encodings) == 1:
        #                 json_path = f"results/brec/ran/{category}_{selected_encodings[0]}_pair_{pair_idx}_statistics.json"
        #                 write_results(f, json_path, pair_results, json_results)

        if not pip:
            total_pair_idx = 0  # Keep track of total pairs processed

            # looping over categories
            for category, graphs in graphs_read_from_files.items():
                if category not in selected_categories:
                    print(f"Skipping {category} category")
                    continue

                print(f"\nProcessing {category} category...")
                num_pairs_to_process = len(graphs) // 2
                print(f"Number of pairs to process: {num_pairs_to_process}")

                for local_pair_idx in range(num_pairs_to_process):

                    if len(selected_encodings) == 1:
                        # check if the json_path = f"results/brec/ran/{category}_pair_{total_pair_idx}_statistics.json"
                        # already exists
                        json_path = f"results/brec/ran/{category}_{selected_encodings[0]}_pair_{total_pair_idx}_statistics.json"
                        if os.path.exists(json_path):
                            print(
                                f"Skipping {category} category as {json_path} already exists"
                            )
                            continue

                    g1 = graphs[local_pair_idx * 2]
                    g2 = graphs[local_pair_idx * 2 + 1]

                    # Skip if either graph has more than 20 nodes
                    # if g1.number_of_nodes() > 20 or g2.number_of_nodes() > 20:
                    #     print(
                    #         f"Skipping pair {local_pair_idx + 1}/5 in {category} "
                    #         f"(too many nodes: G1={g1.number_of_nodes()}, "
                    #         f"G2={g2.number_of_nodes()})"
                    #     )
                    #     continue

                    print(f"(global pair index: {total_pair_idx})")

                    pair_results = analyze_graph_pair(
                        g1,
                        g2,
                        total_pair_idx,  # Use the global pair index
                        category,
                        is_isomorphic=False,
                        already_in_nx=True,
                        types_of_encoding=selected_encodings,
                    )
                    print(
                        f"pair_results: {pair_results['graph_level']['encodings']} and {pair_results['hypergraph_level']['encodings']}"
                    )
                    # Get first key's values using next() and iter()
                    first_encoding_graph = next(
                        iter(pair_results["graph_level"]["encodings"].values())
                    )
                    status_graph = first_encoding_graph["status"]
                    first_encoding_hypergraph = next(
                        iter(pair_results["hypergraph_level"]["encodings"].values())
                    )
                    status_hypergraph = first_encoding_hypergraph["status"]
                    print(f"status_graph: {status_graph}")
                    print(f"status_hypergraph: {status_hypergraph}")
                    if (
                        status_graph != MatchStatus.TIMEOUT
                        and status_hypergraph != MatchStatus.TIMEOUT
                    ):
                        if len(selected_encodings) == 1:
                            json_path = f"results/brec/ran/{category}_{selected_encodings[0]}_pair_{total_pair_idx}_statistics.json"
                            # updates the json_results with the pair_results
                            # and saves the singleton file
                            write_results(f, json_path, pair_results, json_results)
                    else:
                        print("🚨 Timeout")
                        if len(selected_encodings) == 1:
                            json_path = f"results/brec/ran/{category}_{selected_encodings[0]}_pair_{total_pair_idx}_statistics_TIMEOUT.json"
                            # updates the json_results with the pair_results
                            # and saves the singleton file
                            write_results(f, json_path, pair_results, json_results)

                    total_pair_idx += 1  # Increment the global counter

        # Save JSON results with percentages
        # keys are categories
        # this I should be able to recreate from results/brec/ran/
        final_stats: dict = {}
        # loop over categories
        for category in json_results:
            final_stats[category] = {}
            # loop over encodings
            for enc in json_results[category]:
                # Get the statistics for this category-encoding combination
                stats = json_results[category][enc]
                # Calculate the percentage of graphs that were classified as different
                # (number of different graphs / total number of graphs) * 100
                # If no graphs were processed (total = 0), set percentage to 0
                percentage: float = (
                    (stats["different"] / stats["total"]) * 100
                    if stats["total"] > 0
                    else 0
                )
                final_stats[category][enc] = [
                    round(percentage, 2),
                    stats["total"],
                    stats["total_with_timeout"],
                ]

        # all results (all pairs) for the chosen encodings and categories
        # Save JSON file
        with open(json_file, "w") as json_f:
            json.dump(final_stats, json_f, indent=2)

        print(f"\nStatistics saved to: {json_file}")

        # Generate LaTeX table
        generate_latex_table(final_stats)


if __name__ == "__main__":
    # Required for Windows compatibility
    mp.freeze_support()
    main()
