"""Functions for analysing the BREC dataset by category

Should tell us about the distribution of nodes in each category, the counts.
"""

import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def analyze_brec_categories(verbose: bool = False) -> dict:
    """Analyse the BREC dataset by category

    Returns:
        dict: Dictionary mapping categories to lists of NetworkX graphs
    """
    categories: dict = {
        "basic": "basic.npy",
        "regular": "regular.npy",
        "str": "str.npy",  # strongly regular
        "cfi": "cfi.npy",
        "extension": "extension.npy",
        "4vtx": "4vtx.npy",
        "dr": "dr.npy",  # distance regular
    }

    data_path = "BREC_Data"
    print(f"\nLoading data from: {data_path}")
    print("\nBREC Dataset Structure:")
    total_pairs = 0
    total_graphs = 0

    graphs_by_category: dict = {}

    for category, filename in categories.items():
        file_path = os.path.join(data_path, filename)
        try:
            data = np.load(file_path, allow_pickle=True)
            if verbose:
                print(f"\nDEBUG - {category}:")
                print(f"Raw data length: {len(data)}")
                print(
                    f"Data shape (if array): {data.shape if hasattr(data, 'shape') else 'N/A'}"
                )
            # print(f"First element type: {type(data[0])}")

            nx_graphs = []
            if category in ["regular", "cfi", "extension"]:
                if verbose:
                    print(f"DEBUG - {category} special handling:")
                    print(f"Number of pairs in data: {len(data)}")

                # loop through pairs
                for i, pair in enumerate(data):
                    pair_graphs = []
                    try:
                        if category == "extension":
                            for g6_str in pair:
                                G = nx.from_graph6_bytes(g6_str.encode())
                                if not nx.is_connected(G):
                                    if verbose:
                                        print(
                                            f"Skipping disconnected graph in pair {i} of {category}"
                                        )
                                    raise nx.NetworkXError("Graph is not connected")
                                pair_graphs.append(G)
                        else:  # regular and cfi
                            for g6_bytes in pair:
                                G = nx.from_graph6_bytes(g6_bytes)
                                if not nx.is_connected(G):
                                    if verbose:
                                        print(
                                            f"Skipping disconnected graph in pair {i} of {category}"
                                        )
                                    raise nx.NetworkXError("Graph is not connected")
                                pair_graphs.append(G)

                        # If we get here, both graphs in the pair are connected
                        nx_graphs.extend(pair_graphs)

                    except nx.NetworkXError:
                        if verbose:
                            print(f"Skipping pair {i} due to disconnected graph")
                        continue

                print(f"Total connected graphs loaded for {category}: {len(nx_graphs)}")
            else:
                # Handle basic format (alternating graphs)
                assert len(data) % 2 == 0, "Expected even number of pairs for basic"
                for i, g6_str in enumerate(data):
                    try:
                        if isinstance(g6_str, bytes):
                            G = nx.from_graph6_bytes(g6_str)
                        else:
                            G = nx.from_graph6_bytes(g6_str.encode())

                        if not nx.is_connected(G):
                            if verbose:
                                print(f"Skipping disconnected graph {i} in {category}")
                            continue

                        nx_graphs.append(G)
                    except Exception as e:
                        if verbose:
                            print(f"Error loading graph {i} in {category}: {e}")
                        continue

            num_pairs = len(nx_graphs) // 2
            total_pairs += num_pairs
            total_graphs += len(nx_graphs)
            print(f"{category}: {num_pairs} pairs ({len(nx_graphs)} graphs)")
            print(f"Total cumulative: {total_pairs} pairs ({total_graphs} graphs)")

            graphs_by_category[category] = nx_graphs

            # Print info about first graph
            first_graph = nx_graphs[0]
            print(
                f"  First graph: {first_graph.number_of_nodes()} nodes, "
                f"{first_graph.number_of_edges()} edges"
            )

            del nx_graphs

        except Exception as e:
            print(f"Error loading {category}: {e}")
            if len(data) > 0:
                print(f"First item type: {type(data[0])}")
                print(f"First item content: {data[0]}")

    print(f"\nTotal: {total_pairs} pairs ({total_graphs} graphs)")

    return graphs_by_category


def quick_eda_from_github(graphs, verbose: bool = False):
    """
    Note: INPUT ALREADY PROCESSED

    """
    print("\nDEBUG - Starting quick_eda_from_github")

    # Print initial counts
    for category, graph_list in graphs.items():
        if verbose:
            print(f"\nDEBUG 2 - {category}:")
            print(f"Initial graph list length: {len(graph_list)}")
            if len(graph_list) >= 2:
                print(
                    f"First two graphs nodes: {graph_list[0].number_of_nodes()}, {graph_list[1].number_of_nodes()}"
                )

    # Create lists to store node counts for each category
    node_counts: dict = {category: [] for category in graphs.keys()}

    # Collect node counts for each category
    for category, graph_list in graphs.items():
        if verbose:
            print(f"\nDEBUG 3 - Processing {category}:")
            print(f"Category: {category} has {len(graph_list)} graphs")
        for G in graph_list:
            node_counts[category].append(G.number_of_nodes())

        print(f"Final node count list length: {len(node_counts[category])}")
        if len(node_counts[category]) >= 2:
            print(
                f"First two node counts: {node_counts[category][0]}, {node_counts[category][1]}"
            )

    # Create histogram
    plt.figure(figsize=(12, 6))

    # Plot histogram for each category with different colors
    colors = ["skyblue", "lightgreen", "salmon", "lightgray", "wheat"]
    for (category, counts), color in zip(node_counts.items(), colors):
        if counts:  # Only plot if category has graphs
            plt.hist(
                counts,
                bins="auto",
                alpha=0.5,
                label=category,
                color=color,
                edgecolor="black",
            )

    plt.xlabel("Number of Nodes", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Node Counts Across BREC Graph Categories", fontsize=14)
    plt.legend()

    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    total_count = 0

    # Add text box with statistics for each category
    stats_text = ""
    for category, counts in node_counts.items():
        if counts:  # Only add stats if category has graphs
            stats_text += f"{category}:\n"
            total_count += len(counts)
            stats_text += f"  Pairs: {len(counts)/2}\n"  # These are the unique graphs
            stats_text += f"  Count: {len(counts)}\n"
            stats_text += f"  Total Count: {total_count}\n"
    plt.text(
        1.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Save plot
    plt.tight_layout()
    plt.savefig("plots/brec_node_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Print original analysis
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
