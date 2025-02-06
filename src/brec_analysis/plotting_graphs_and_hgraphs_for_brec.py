"""Plotting functions for the BREC dataset"""

import os

import hypernetx as hnx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.figure import Figure

from brec_analysis.utils_for_brec import create_comparison_table


def plot_hypergraph_pair(
    G1: nx.Graph,
    G2: nx.Graph,
    hg1: dict,
    hg2: dict,
    pair_idx: str,
    category: str,
    is_isomorphic: bool,
    output_dir: str,
):
    """Plot comparison of two hypergraphs with their bipartite representations.

    Args:
        G1 (nx.Graph):
            The first graph.
        G2 (nx.Graph):
            The second graph.
        hg1 (dict):
            The first hypergraph.
        hg2 (dict):
            The second hypergraph.
        pair_idx (str):
            The index of the pair.
        category (str):
            The category of the pair.
        is_isomorphic (bool):
            Whether the graphs are isomorphic.
        output_dir (str):
            The directory to save the plots.
    """
    # Create figure with 4x2 subplot grid (increased height for new row)
    plt.figure(figsize=(35, 15))
    plt.suptitle(f"Pair {pair_idx} ({category})", fontsize=35)

    # Row 1: Original graphs
    # Plot first graph
    plt.subplot(241)
    pos1 = nx.circular_layout(G1)
    plt.title(f"Graph A\n{len(G1.nodes())} nodes, {len(G1.edges())} edges", fontsize=30)
    nx.draw(
        G1,
        pos1,
        node_color="lightblue",
        node_size=500,
        with_labels=True,
        font_size=15,
        font_weight="bold",
    )

    # Plot second graph
    plt.subplot(242)
    pos2 = nx.circular_layout(G2)
    plt.title(f"Graph B\n{len(G2.nodes())} nodes, {len(G2.edges())} edges", fontsize=30)
    nx.draw(
        G2,
        pos2,
        node_color="lightpink",
        node_size=500,
        with_labels=True,
        font_size=15,
        font_weight="bold",
    )
    ####################

    # Row 2: Hypergraph visualizations
    # Plot first hypergraph
    plt.subplot(243)
    H1 = hnx.Hypergraph(hg1["hypergraph"])
    hnx.draw(
        H1,
        pos=pos1,
        node_radius=2,
        node_labels_kwargs={"fontsize": 15},
        with_node_labels=True,
        with_edge_labels=False,
        convex=False,
    )
    plt.title(f"Hypergraph A\n({len(hg1['hypergraph'])} hyperedges)", fontsize=30)

    # Plot second hypergraph
    plt.subplot(244)
    H2 = hnx.Hypergraph(hg2["hypergraph"])
    hnx.draw(
        H2,
        pos=pos2,
        node_radius=2,
        node_labels_kwargs={"fontsize": 15},
        with_node_labels=True,
        with_edge_labels=False,
        convex=False,
    )
    plt.title(f"Hypergraph B\n({len(hg2['hypergraph'])} hyperedges)", fontsize=30)

    # # Row 3: Bipartite representations
    # # Plot first bipartite
    # plt.subplot(245)
    # BH1 = H1.bipartite()
    # top1 = set(n for n, d in BH1.nodes(data=True) if d["bipartite"] == 0)
    # pos1 = nx.bipartite_layout(BH1, top1)
    # nx.draw(
    #     BH1,
    #     pos1,
    #     with_labels=True,
    #     node_color=[
    #         "lightblue" if node in top1 else "lightgreen" for node in BH1.nodes()
    #     ],
    #     node_size=500,
    #     font_size=12,
    #     font_weight="bold",
    # )
    # plt.title("Graph A Bipartite")

    # # Plot second bipartite
    # plt.subplot(245)
    # BH2 = H2.bipartite()
    # top2 = set(n for n, d in BH2.nodes(data=True) if d["bipartite"] == 0)
    # pos2 = nx.bipartite_layout(BH2, top2)
    # nx.draw(
    #     BH2,
    #     pos2,
    #     with_labels=True,
    #     node_color=[
    #         "lightblue" if node in top2 else "lightgreen" for node in BH2.nodes()
    #     ],
    #     node_size=500,
    #     font_size=12,
    #     font_weight="bold",
    # )
    # plt.title("Bipartite B")

    # Row 4: Hyperedge size distributions
    # Plot first histogram
    plt.subplot(245)
    sizes1 = [len(edge) for edge in hg1["hypergraph"].values()]
    plt.hist(
        sizes1,
        bins=range(min(sizes1), max(sizes1) + 2),
        alpha=0.7,
        color="lightblue",
        rwidth=0.8,
    )
    plt.title(
        f"Hypergraph A Edge Degree \n Distribution({len(hg1['hypergraph'])} hyperedges)",
        fontsize=30,
    )
    plt.xlabel("Hyperedge Size", fontsize=30)
    plt.ylabel("Count", fontsize=30)
    plt.grid(True, alpha=0.3)
    # Set x-axis ticks to integers only
    plt.xticks(range(min(sizes1), max(sizes1) + 1))
    plt.tick_params(axis="both", which="major", labelsize=20)  # Add this line

    # Plot second histogram
    plt.subplot(246)
    sizes2 = [len(edge) for edge in hg2["hypergraph"].values()]
    plt.hist(
        sizes2,
        bins=range(min(sizes2), max(sizes2) + 2),
        alpha=0.7,
        color="lightpink",
        rwidth=0.8,
    )
    plt.title(
        f"Hypergraph B Edge Degree \n Distribution({len(hg2['hypergraph'])} hyperedges)",
        fontsize=30,
    )
    plt.xlabel("Hyperedge Size", fontsize=30)
    plt.ylabel("Count", fontsize=30)
    plt.grid(True, alpha=0.3)
    # Set x-axis ticks to integers only
    plt.xticks(range(min(sizes2), max(sizes2) + 1))
    plt.tick_params(axis="both", which="major", labelsize=20)  # Add this line

    #########################################################################################

    # Row 4: Node degree distributions (number of hyperedges each node belongs to)
    plt.subplot(247)
    # Calculate node degrees for first hypergraph
    node_degrees1: dict = {}
    for edge in hg1["hypergraph"].values():
        for node in edge:
            node_degrees1[node] = node_degrees1.get(node, 0) + 1
    degrees1 = list(node_degrees1.values())

    plt.hist(
        degrees1,
        bins=range(min(degrees1), max(degrees1) + 2),
        alpha=0.7,
        color="lightblue",
        rwidth=0.8,
    )
    plt.title(
        "Hypergraph A Node Degree \n Distribution",
        fontsize=30,
    )
    plt.xlabel("Node Degree", fontsize=30)
    plt.ylabel("Count", fontsize=30)
    plt.grid(True, alpha=0.3)
    # Set x-axis ticks to integers only
    plt.xticks(range(min(degrees1), max(degrees1) + 1))
    plt.tick_params(axis="both", which="major", labelsize=20)  # Add this line

    plt.subplot(248)

    # Calculate node degrees for second hypergraph
    node_degrees2: dict = {}
    for edge in hg2["hypergraph"].values():
        for node in edge:
            node_degrees2[node] = node_degrees2.get(node, 0) + 1
    degrees2 = list(node_degrees2.values())

    plt.hist(
        degrees2,
        bins=range(min(degrees2), max(degrees2) + 2),
        alpha=0.7,
        color="lightpink",
        rwidth=0.8,
    )
    plt.title(
        "Hypergraph B Node Degree \n Distributions",
        fontsize=30,
    )
    plt.xlabel("Node Degree", fontsize=30)
    plt.ylabel("Count", fontsize=30)
    plt.grid(True, alpha=0.3)
    # Set x-axis ticks to integers only
    plt.xticks(range(min(degrees2), max(degrees2) + 1))
    plt.tick_params(axis="both", which="major", labelsize=20)  # Add this line

    plt.tight_layout()
    os.makedirs("plots/hypergraphs", exist_ok=True)
    plt.savefig(
        f"plots/hypergraphs/pair_{pair_idx}_{category.lower()}_hypergraphs.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # Second figure: Statistics
    plt.figure(figsize=(10, 6))
    plt.title(f"Hypergraph Statistics - Pair {pair_idx} ({category})", pad=20)
    plt.axis("off")

    # Convert lists to tuples for hashing
    hyperedge_sizes1 = set(tuple(v) for v in hg1["hypergraph"].values())
    hyperedge_sizes2 = set(tuple(v) for v in hg2["hypergraph"].values())

    stats_text = [
        f"Graph A: {len(hg1['hypergraph'])} hyperedges",
        f"Graph B: {len(hg2['hypergraph'])} hyperedges",
        "\nHyperedge sizes Graph A:",
        *[
            f"Size {len(v)}: {sum(1 for e in hg1['hypergraph'].values() if len(e) == len(v))}"
            for v in hyperedge_sizes1
        ],
        "\nHyperedge sizes Graph B:",
        *[
            f"Size {len(v)}: {sum(1 for e in hg2['hypergraph'].values() if len(e) == len(v))}"
            for v in hyperedge_sizes2
        ],
    ]
    plt.text(
        0.1,
        0.5,
        "\n".join(stats_text),
        fontsize=12,
        transform=plt.gca().transAxes,
    )

    plt.tight_layout()
    plt.savefig(
        f"plots/hypergraphs/pair_{pair_idx}_{category.lower()}_statistics.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def plot_graph_pair(
    graph1: nx.Graph,
    graph2: nx.Graph,
    pair_idx: str,
    category: str,
    is_isomorphic: bool,
    output_dir: str,
) -> None:
    """Plot a pair of graphs side by side with graph statistics and degree distributions.

    Args:
        graph1 (nx.Graph):
            The first graph.
        graph2 (nx.Graph):
                The second graph.
        pair_idx (str):
            The index of the pair.
        category (str):
            The category of the pair.
        is_isomorphic (bool):
            Whether the graphs are isomorphic.
        output_dir (str):
            The directory to save the plots.
    """
    # Create figure with 3x2 subplot grid (added row for adjacency matrices)
    fig: Figure = plt.figure(figsize=(16, 32))

    # Set isomorphism status

    # Plot first graph
    ax1 = plt.subplot(3, 2, 1)
    pos1 = nx.circular_layout(graph1)
    plt.title(
        f"Graph A\n{len(graph1.nodes())} nodes, {len(graph1.edges())} edges",
        fontsize=25,
    )
    nx.draw(
        graph1,
        pos1,
        node_color="lightblue",
        node_size=500,
        with_labels=True,
        font_size=10,
        font_weight="bold",
    )

    # Plot second graph
    ax2 = plt.subplot(3, 2, 2)
    pos2 = nx.circular_layout(graph2)
    plt.title(
        f"Graph B\n{len(graph2.nodes())} nodes, {len(graph2.edges())} edges",
        fontsize=25,
    )
    nx.draw(
        graph2,
        pos2,
        node_color="lightpink",
        node_size=500,
        with_labels=True,
        font_size=10,
        font_weight="bold",
    )

    # Plot degree distributions
    ax3 = plt.subplot(3, 2, 3)
    degrees1 = [d for n, d in graph1.degree()]
    degrees2 = [d for n, d in graph2.degree()]
    max_degree = max(max(degrees1), max(degrees2))
    min_degree = min(min(degrees1), min(degrees2))
    bins = range(min_degree, max_degree + 2)  # +2 to include max degree

    # Set the width and positions for the bars
    width = 0.35  # Width of the bars
    x = np.array(list(bins[:-1]))  # Bar positions for graph1

    # Create the bars with offset positions
    plt.bar(
        x - width / 2,
        np.histogram(degrees1, bins=bins)[0],
        width,
        alpha=0.7,
        color="lightblue",
        label="Graph A",
    )
    plt.bar(
        x + width / 2,
        np.histogram(degrees2, bins=bins)[0],
        width,
        alpha=0.7,
        color="lightpink",
        label="Graph B",
    )

    plt.title(
        f"Node Degree Distribution\n({len(graph1.nodes())} total nodes)", fontsize=25
    )
    plt.xlabel("Degree", fontsize=25)
    plt.ylabel("Count", fontsize=25)
    plt.grid(True, alpha=0.3)
    plt.legend(prop={"size": 25})
    # Set x-axis ticks to integers only
    plt.xticks(range(min_degree, max_degree + 1))

    # # Add text with exact counts for Graph A
    # unique_degrees1 = sorted(set(degrees1))
    # degree_counts1 = {deg: degrees1.count(deg) for deg in unique_degrees1}
    # text1 = "Graph A:\n" + "\n".join(
    #     [f"Degree {deg}: {count}" for deg, count in degree_counts1.items()]
    # )
    # plt.text(
    #     0.95,
    #     0.95,
    #     text1,
    #     transform=ax3.transAxes,
    #     verticalalignment="top",
    #     horizontalalignment="right",
    #     bbox=dict(facecolor="white", alpha=0.8),
    #     fontsize=25,
    # )

    # Plot second histogram
    ax3.hist(
        degrees2,
        bins=bins,
        alpha=0.7,
        color="lightpink",
        rwidth=0.8,
        label="Graph B",
    )

    # # Add text with exact counts for Graph B
    # unique_degrees2 = sorted(set(degrees2))
    # degree_counts2 = {deg: degrees2.count(deg) for deg in unique_degrees2}
    # text2 = "Graph B:\n" + "\n".join(
    #     [f"Degree {deg}: {count}" for deg, count in degree_counts2.items()]
    # )
    # plt.text(
    #     0.75,
    #     0.95,
    #     text2,
    #     transform=ax3.transAxes,
    #     verticalalignment="top",
    #     horizontalalignment="right",
    #     bbox=dict(facecolor="white", alpha=0.8),
    #     fontsize=25,
    # )

    # Compute and display graph statistics
    ax4 = plt.subplot(3, 2, 4)
    ax4.axis("off")

    def get_graph_stats(G):
        stats = {
            # Basic statistics
            "Number of nodes": G.number_of_nodes(),
            "Number of edges": G.number_of_edges(),
            "Average degree": np.mean([d for n, d in G.degree()]),
            "Maximum degree": max([d for n, d in G.degree()]),
            "Minimum degree": min([d for n, d in G.degree()]),
            "Density": nx.density(G),
            # Structural properties
            "Is bipartite": nx.is_bipartite(G),
            # "Number of triangles": sum(nx.triangles(G).values()) // 3,
            # Convert generator to list to count cliques
            # number of maximal cliques
            "Number of maximal cliques": sum(1 for c in nx.find_cliques(G)),
            # the largest maximal clique
            # "Largest maximal clique": max(nx.find_cliques(G), key=len),
            "Edge connectivity": nx.edge_connectivity(G),
            "Node connectivity": nx.node_connectivity(G),
            "Number of components": nx.number_connected_components(G),
            "Is planar": nx.is_planar(G),
            # Centrality measures (averaged over nodes)
            "Avg betweenness": np.mean(list(nx.betweenness_centrality(G).values())),
            "Avg closeness": np.mean(list(nx.closeness_centrality(G).values())),
            # "Avg eigenvector": np.mean(
            #     list(nx.eigenvector_centrality_numpy(G).values())
            # ),
            # Spectral properties
            # "Spectral radius": max(abs(nx.adjacency_spectrum(G))),
            "Algebraic connectivity": nx.algebraic_connectivity(G),
            # "Spectral gap": sorted(abs(nx.adjacency_spectrum(G)))[-1]
            # - sorted(abs(nx.adjacency_spectrum(G)))[-2],
        }

        # Add these stats only if graph is connected
        if nx.is_connected(G):
            stats.update(
                {
                    "Diameter": nx.diameter(G),
                    "Average shortest path": nx.average_shortest_path_length(G),
                    "Average clustering": nx.average_clustering(G),
                    # "Assortativity": nx.degree_assortativity_coefficient(G),
                    "Radius": nx.radius(G),
                    "Center size": len(nx.center(G)),
                    "Periphery size": len(nx.periphery(G)),
                }
            )
            # try:
            #     stats["Girth"] = (
            #         len(min(nx.cycle_basis(G), key=len))
            #         if nx.cycle_basis(G)
            #         else float("inf")
            #     )
            # except:
            #     stats["Girth"] = "N/A"
        # else:
        #     stats.update(
        #         {
        #             "Diameter": "N/A (disconnected)",
        #             "Average shortest path": "N/A (disconnected)",
        #             "Average clustering": nx.average_clustering(G),
        #             "Assortativity": nx.degree_assortativity_coefficient(G),
        #             "Girth": "N/A (disconnected)",
        #             "Radius": "N/A (disconnected)",
        #             "Center size": "N/A (disconnected)",
        #             "Periphery size": "N/A (disconnected)",
        #         }
        #     )

        return stats

    stats1 = get_graph_stats(graph1)
    stats2 = get_graph_stats(graph2)

    # Create comparison table with colored differences
    table_text, colors = create_comparison_table(stats1, stats2)

    # Plot statistics with colored differences - smaller text and spacing
    y_pos = 0.98
    line_height = 0.025  # Reduced from 0.04

    # Plot title
    ax4.text(
        0.05,
        y_pos,
        "Graph Statistics: \n",
        fontsize=16,  # Reduced from 10
        family="monospace",
        verticalalignment="top",
        transform=ax4.transAxes,
        color="black",
        fontweight="bold",
    )

    # Plot each statistic with appropriate color
    y_pos -= line_height  # Reduced space after title
    for text, color in zip(table_text, colors):
        ax4.text(
            0.05,
            y_pos,
            text,
            fontsize=15,
            family="DejaVu Sans",  # Changed from "monospace" to a nicer font
            verticalalignment="top",
            transform=ax4.transAxes,
            color=color,
        )
        y_pos -= line_height

    # Add main title
    plt.suptitle(
        f"BREC Dataset - {category} Category\nPair {pair_idx}",
        fontsize=25,
        y=1.02,
    )

    # Add adjacency matrix plots in new row
    ax5 = plt.subplot(3, 2, 5)
    adj1 = nx.adjacency_matrix(graph1).todense()
    adj2 = nx.adjacency_matrix(graph2).todense()

    # Plot first adjacency matrix
    im1 = ax5.imshow(adj1, cmap="viridis")
    plt.colorbar(im1, ax=ax5, shrink=0.8)
    ax5.set_title("Graph A Adjacency Matrix", fontsize=25)
    ax5.set_xlabel("Node Index", fontsize=25)
    ax5.set_ylabel("Node Index", fontsize=25)

    # Plot second adjacency matrix
    ax6 = plt.subplot(3, 2, 6)
    im2 = ax6.imshow(adj2, cmap="viridis")
    plt.colorbar(im2, ax=ax6, shrink=0.8)
    ax6.set_title("Graph B Adjacency Matrix", fontsize=25)
    ax6.set_xlabel("Node Index", fontsize=25)
    ax6.set_ylabel("Node Index", fontsize=25)

    # Create separate figure for difference plot
    plt.figure(figsize=(8, 6))
    # Compute difference (padding smaller matrix if sizes differ)
    max_size = max(adj1.shape[0], adj2.shape[0])
    padded1 = np.pad(
        adj1, ((0, max_size - adj1.shape[0]), (0, max_size - adj1.shape[0]))
    )
    padded2 = np.pad(
        adj2, ((0, max_size - adj2.shape[0]), (0, max_size - adj2.shape[0]))
    )
    diff = padded1 - padded2

    im3 = plt.imshow(diff, cmap="Blues", vmin=-1, vmax=1)
    plt.colorbar(im3, shrink=0.8)
    plt.title("Adjacency Matrix Difference (A - B)")
    plt.xlabel("Node Index")
    plt.ylabel("Node Index")

    # Save difference plot
    plt.tight_layout()
    os.makedirs(f"{output_dir}/adjacency_diffs", exist_ok=True)
    plt.savefig(
        f"{output_dir}/adjacency_diffs/pair_{pair_idx}_{category.lower()}_diff.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # Return to main figure and finish
    plt.figure(fig.number)  # type: ignore
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/pair_{pair_idx}_{category.lower()}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
