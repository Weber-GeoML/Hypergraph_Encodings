"""
This script is used to study the theorems.

Load example graphs and computes the encodings.

OR curvature can distinguish graph pairs that are 3-WL indistinguishable witjout needing more than 1-hop of neighborhood information.

I - We want graphs that are non-isomorphic, but not distinguishable with classical MPGNNs.
II - We would also want graphs that are non-isomorphic, but that are isospectral, that cannot be distinguisable with classical MPGNNS, and that cannot be distinguised with the LAPE encoding but can be distinguised with other encoding.
III -Then we would like this:
"Message-Passing GNNs (MPGNNs) with H-LAPE are strictly more expressive
than the 1-WL test and hence than MPGNNs without encodings. Further, there exist graphs which can be distinguished
using H-LAPE, but not using standard, graph-level LAPE."
-> so we need non-isomoprhic graphs, with the same LAPE encodings (isospectral), but that can be distingished with H-LAPE.
IV - MPGNNs with H-RWPE are strictly more expressive than the 1-WL test and
hence than MPGNNs without encodings. There exist graphs
which can be distinguished using H-RWPE, but not using graph-level RWPE.
-> so we need non-isomoprhic graphs, with the same RWPE encodings, but that can be distingished with H-RWPE.

First example:
the 4x4 Rooke and Shrikhande graphs are non-isomorphic, and their spectra are identical.
LDP encodings cannot distinguish them either, as they have identical node degree distributions. While
nodes in both graphs have identical node degrees (e.g., could not be distinguished with classical
MPGNNs), their 2-hop connectivities differ.


What we are going to do in code:
find exmaple of these graphs and load them.
compute the encodings using the exesting code.
Plot the following: degree distribution, and others

Answer:
I - Rooke and Shrikhande graphs. Non-isomorphic, cannot be distingushed with classical MPGNNs.
II - Rooke and Shrikhande graphs: non-isomoprhic, isospectral, not distinguishabel with LDP, not distinguishabel with LAPE, distinguishabel with curvature.
H-LAPE
III - Try the Rooke and Shrikhande graphs with H-LAPE: todo
Here is what I can do: grab these graphs. Then make the hypergraphs as follow:
turn triangles into hyperedges.

House of graphs is a great place to find the graphs!

TODO:
- turn this into a script that takes any two graphs and tells which encodings are the same and which are different.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from brec_analysis.check_encodings_same import (check_isospectrality,
                                                checks_encodings,
                                                matrix_to_pmatrix,
                                                reconstruct_matrix)
from encodings_hnns.encodings import HypergraphEncodings


def test_laplacian(
    hg1,
    hg2,
    lap_type,
    name1: str = "Graph1",
    name2: str = "Graph2",
    verbose: bool = False,
):
    """
    Args:
        hg1, hg2: hypergraphs
        lap_type: type of Laplacian to use
        name1, name2: names of the graphs
        verbose: whether to print verbose output
    Returns:
        hg1_lape, hg2_lape: LAPE encodings of the two graphs
        laplacian_1, laplacian_2: Laplacian matrices of the two graphs
        same: whether the two graphs are the same
    TODO: to finish! Should be easy and fast now.
    """
    if verbose:
        print(f"Testing Laplacian type: {lap_type}")
    # Initialize encoder
    encoder_shrikhande = HypergraphEncodings()
    encoder_rooke = HypergraphEncodings()

    hg1_lape = encoder_shrikhande.add_laplacian_encodings(
        hg1.copy(), laplacian_type=lap_type, verbose=False, use_same_sign=True
    )
    hg2_lape = encoder_rooke.add_laplacian_encodings(
        hg2.copy(), laplacian_type=lap_type, verbose=False, use_same_sign=True
    )

    # Get the appropriate Laplacian matrix based on type
    if lap_type == "Normalized":
        laplacian_1 = encoder_shrikhande.laplacian.normalized_laplacian
        laplacian_2 = encoder_rooke.laplacian.normalized_laplacian
        degree_vertices1 = encoder_shrikhande.laplacian.degree_vertices
        degree_vertices2 = encoder_rooke.laplacian.degree_vertices
        assert np.allclose(
            degree_vertices1,
            6 * np.eye(degree_vertices1.shape[0]),
            atol=1e-12,
            rtol=1e-12,
        ), "degree_vertices is not the identity matrix"
        assert np.allclose(
            degree_vertices2,
            6 * np.eye(degree_vertices2.shape[0]),
            atol=1e-12,
            rtol=1e-12,
        ), "degree_vertices is not the identity matrix"
    elif lap_type == "RW":
        laplacian_1 = encoder_shrikhande.laplacian.rw_laplacian
        laplacian_2 = encoder_rooke.laplacian.rw_laplacian
    else:  # Hodge
        laplacian_1 = encoder_shrikhande.laplacian.hodge_laplacian_up
        laplacian_2 = encoder_rooke.laplacian.hodge_laplacian_up
        laplacian_3 = encoder_shrikhande.laplacian.hodge_laplacian_down
        laplacian_4 = encoder_rooke.laplacian.hodge_laplacian_down

    # print the norms of the laplacians eigenvectors
    eigenvalues_shrikhande, eigenvectors_shrikhande = np.linalg.eigh(laplacian_1)
    eigenvalues_rooke, eigenvectors_rooke = np.linalg.eigh(laplacian_2)

    # print the eigenvalues
    # print(f"Eigenvalues of Shrikhande Laplacian: {eigenvalues_shrikhande}")
    # print(f"Eigenvalues of Rooke Laplacian: {eigenvalues_rooke}")
    # assert they are in order. Smaller to bigger.
    assert np.allclose(
        eigenvalues_shrikhande, np.sort(eigenvalues_shrikhande)
    ), "Eigenvalues of Shrikhande Laplacian are not in order"
    assert np.allclose(
        eigenvalues_rooke, np.sort(eigenvalues_rooke)
    ), "Eigenvalues of Rooke Laplacian are not in order"

    if lap_type == "Normalized" or lap_type == "Hodge":

        # assert that laplacian_1 and laplacian_2 are symmetric
        assert np.allclose(
            laplacian_1, laplacian_1.T, atol=1e-12, rtol=1e-12
        ), "laplacian_1 is not symmetric"
        assert np.allclose(
            laplacian_2, laplacian_2.T, atol=1e-12, rtol=1e-12
        ), "laplacian_2 is not symmetric"

        # Can already the min value, max value and other statistics
        # to check wether the encodings are the same or not!!!!
        # TODO!
        min_shrikhande = np.min(eigenvectors_shrikhande)
        min_rooke = np.min(eigenvectors_rooke)
        if verbose:
            # print the min value in the eigenvectors
            print(f"Min value in eigenvectors of Shrikhande: {min_shrikhande}")
            print(f"Min value in eigenvectors of Rooke: {min_rooke}")
        # ranks
        rank_shrikhande = np.linalg.matrix_rank(eigenvectors_shrikhande)
        rank_rooke = np.linalg.matrix_rank(eigenvectors_rooke)
        # print the rank of the eigenvectors
        print(f"Rank of eigenvectors of Shrikhande: {rank_shrikhande}")
        print(f"Rank of eigenvectors of Rooke: {rank_rooke}")
        if rank_shrikhande != rank_rooke:
            print("The two graphs have different encodings")
            return hg1_lape, hg2_lape, laplacian_1, laplacian_2, False
        if min_shrikhande != min_rooke:
            print("The two graphs have different encodings")
            return hg1_lape, hg2_lape, laplacian_1, laplacian_2, False

        # assert that the matrix of eigenvectors is orthonormal
        assert np.allclose(
            eigenvectors_shrikhande @ eigenvectors_shrikhande.T,
            np.eye(eigenvectors_shrikhande.shape[0]),
            atol=1e-12,
            rtol=1e-12,
        ), f"Eigenvectors of Shrikhande are not orthonormal. Difference: {eigenvectors_shrikhande.T @ eigenvectors_shrikhande - np.eye(eigenvectors_shrikhande.shape[0])}"
        assert np.allclose(
            eigenvectors_rooke @ eigenvectors_rooke.T,
            np.eye(eigenvectors_rooke.shape[0]),
            atol=1e-12,
            rtol=1e-12,
        ), f"Eigenvectors of Rooke are not orthonormal. Difference: {eigenvectors_rooke.T @ eigenvectors_rooke - np.eye(eigenvectors_rooke.shape[0])}"

        # Define matrices and their properties to check
        matrices_to_check = [
            (
                "Shrikhande",
                laplacian_1,
                eigenvalues_shrikhande,
                eigenvectors_shrikhande,
            ),
            ("Rooke", laplacian_2, eigenvalues_rooke, eigenvectors_rooke),
        ]

        # Store properties for comparison
        properties: dict[str, dict[str, float | np.ndarray]] = {
            name: {} for name, *_ in matrices_to_check
        }

        # Check all properties for each matrix
        for (
            name,
            original_matrix,
            eigenvalues,
            eigenvectors,
        ) in matrices_to_check:
            # Check symmetry
            assert np.allclose(
                original_matrix, original_matrix.T, atol=1e-12, rtol=1e-12
            ), f"{name} matrix is not symmetric"

            # Store basic properties
            properties[name].update(
                {
                    "min_eigenvector": np.min(eigenvectors),
                    "rank": np.linalg.matrix_rank(eigenvectors),
                    "norms": np.sort(np.linalg.norm(eigenvectors, axis=1)),
                }
            )

            if verbose:
                print(f"\nProperties for {name}:")
                print(
                    f"Min value in eigenvectors: {properties[name]['min_eigenvector']}"
                )
                print(f"Rank of eigenvectors: {properties[name]['rank']}")
                print(f"Sorted norms of eigenvectors: {properties[name]['norms']}")

            # Check orthonormality
            orthonormal_diff = eigenvectors @ eigenvectors.T - np.eye(
                eigenvectors.shape[0]
            )
            assert np.allclose(
                eigenvectors @ eigenvectors.T,
                np.eye(eigenvectors.shape[0]),
                atol=1e-12,
                rtol=1e-12,
            ), f"Eigenvectors of {name} are not orthonormal. Difference: {orthonormal_diff}"

            # Check matrix reconstruction
            reconstructed_matrix = reconstruct_matrix(eigenvalues, eigenvectors)
            if np.allclose(
                reconstructed_matrix, original_matrix, atol=1e-12, rtol=1e-12
            ):
                print(f"{name} reconstructed matrix is close to the original matrix.")
                print("Symmetric matrix. Expected for Hodge, normalized")
            else:
                print(f"{name} reconstructed matrix differs from the original matrix.")
                print(f"Difference matrix:\n{reconstructed_matrix - original_matrix}")
                max_diff = np.max(np.abs(reconstructed_matrix - original_matrix))
                print(f"Maximum difference: {max_diff}")
                assert False, f"{name} matrix reconstruction failed"

        # Compare properties between graphs
        for prop in ["min_eigenvector", "rank"]:
            if properties["Shrikhande"][prop] != properties["Rooke"][prop]:
                print(f"The two graphs have different encodings (different {prop})")
                return hg1_lape, hg2_lape, laplacian_1, laplacian_2, False

        # Compare norms
        if not np.allclose(
            properties["Shrikhande"]["norms"],
            properties["Rooke"]["norms"],
            atol=1e-12,
            rtol=1e-12,
        ):
            print(
                "The two graphs have different encodings (different eigenvector norms)"
            )
            return hg1_lape, hg2_lape, laplacian_1, laplacian_2, False

        print("\nComparison of eigenvector norms:")
        for name in ["Shrikhande", "Rooke"]:
            print(f"{name} Laplacian eigenvector norms: {properties[name]['norms']}")

    norms_shrikhande = np.linalg.norm(eigenvectors_shrikhande, axis=1)
    norms_rooke = np.linalg.norm(eigenvectors_rooke, axis=1)

    sorted_norms_shrikhande = np.sort(norms_shrikhande)
    sorted_norms_rooke = np.sort(norms_rooke)
    # print the norms of the eigenvectors
    print(
        f"Norm of the eigenvectors of Shrikhande Laplacian: {sorted_norms_shrikhande}"
    )
    print(f"Norm of the eigenvectors of Rooke Laplacian: {sorted_norms_rooke}")
    # print the encodings
    if True:
        # print(f"Shrikhande LAPE encodings: {hg1_lape['features']}")
        # print(f"Rooke LAPE encodings: {hg2_lape['features']}")
        # print the diffs
        # print(f"Diff of the two LAPE encodings: {np.abs(hg1_lape['features'] - hg2_lape['features'])}")
        # print the maximum difference
        print(
            f"Maximum difference in features: {np.max(np.abs(hg1_lape['features'] - hg2_lape['features']))}"
        )

    # Save the heatmap of the difference of the features
    plt.imshow(
        hg1_lape["features"] - hg2_lape["features"],
        cmap="hot",
        interpolation="nearest",
    )
    plt.colorbar()
    plt.title(f"Difference in {lap_type} Features")
    plt.savefig(f"diff_features_{lap_type.lower()}.png")
    plt.close()

    # save the diff of the laplacians as a matrix heatmap
    plt.imshow(laplacian_1 - laplacian_2, cmap="Blues", interpolation="nearest")
    plt.colorbar()
    plt.title(f"Difference in {lap_type} Laplacian Matrices")
    plt.savefig(f"diff_laplacian_{lap_type.lower()}.png")
    plt.close()

    if lap_type == "Hodge":
        # save the diff of the laplacians as a matrix heatmap
        plt.imshow(laplacian_3 - laplacian_4, cmap="Blues", interpolation="nearest")
        plt.colorbar()
        plt.title(f"Difference in {lap_type} Down-Laplacian Matrices")
        plt.savefig(f"diff_laplacian_down_{lap_type.lower()}.png")
        plt.close()

    # print the laplacians
    if False:
        print(f"Shrikhande LAPE laplacian: {laplacian_1}")
        print(f"Rooke LAPE laplacian: {laplacian_2}")

    assert (
        laplacian_1.shape == laplacian_2.shape
    ), "The two laplacians have different shapes"

    # loop through row by row
    for i in range(laplacian_1.shape[0]):
        # print(f"Row {i} of Shrikhande LAPE laplacian: {laplacian_1[i]}")
        # print(f"Row {i} of Rooke LAPE laplacian: {laplacian_2[i]}")
        print(f"Diff of row {i}: {np.abs(laplacian_1[i] - laplacian_2[i])}")
    # loop through column by column
    for i in range(laplacian_1.shape[1]):
        print(f"Diff of column {i}: {np.abs(laplacian_1[:,i] - laplacian_2[:,i])}")

    with open(f"shrikhande_laplacian_{lap_type.lower()}.tex", "w") as f:
        f.write(matrix_to_pmatrix(laplacian_1))

    with open(f"rooke_laplacian_{lap_type.lower()}.tex", "w") as f:
        f.write(matrix_to_pmatrix(laplacian_2))

    # The Laplacians are different!
    if False:
        # print the max difference
        print(f"Max difference: {np.max(np.abs(laplacian_1 - laplacian_2))}")
        # print the difference matrix
        print(f"Difference matrix: {laplacian_1 - laplacian_2}")

        # assert that the two laplacians are the same
        assert np.allclose(
            laplacian_1, laplacian_2, rtol=1e-12, atol=1e-12
        ), "The two laplacians are not the same"

    # compute the eigenvalues
    eigenvalues_shrikhande, eigenvectors_shrikhande = np.linalg.eig(laplacian_1)
    eigenvalues_rooke, eigenvectors_rooke = np.linalg.eig(laplacian_2)
    are_isospectral = check_isospectrality(eigenvalues_shrikhande, eigenvalues_rooke)
    assert are_isospectral, "The two graphs are not isospectral"

    # TODO: we will do the same but at the hypegraph level.

    assert (
        hg1_lape["features"].shape == hg2_lape["features"].shape
    ), "The two LAPE encodings have different shapes"
    print("*" * 100)
    return hg1_lape, hg2_lape, laplacian_1, laplacian_2, True


def print_graph_properties(G):
    """Prints the properties of the graph.

    Args:
        G:
        the graph


    """
    # Print number of nodes and edges
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
    # # Chromatic number 4
    # print("Chromatic number:", nx.chromatic_number(G))
    # diameter 2
    print("Diameter:", nx.diameter(G))
    # Eulerian	yes
    print("Eulerian:", nx.is_eulerian(G))
    # girth	3
    print("Girth:", nx.girth(G))
    # radius	2
    print("Radius:", nx.radius(G))
    # regular	yes
    print("Regular:", nx.is_regular(G))
    print("*" * 100)


# Using the relevant code, compute the encodings:


def convert_nx_to_hypergraph_dict(G):
    """Convert NetworkX graph to hypergraph dictionary format.

    Args:
        G:
            the graph

    Returns:
        the hypergraph dictionary
    """
    print(f"Graph: {G}")
    # Create edge dictionary where each edge is a hyperedge of size 2
    hyperedges = {f"e_{i}": list(edge) for i, edge in enumerate(G.edges())}

    # Create features matrix (initially empty)
    n = G.number_of_nodes()
    features = np.empty((n, 0))

    print(f"Hypergraph: {hyperedges}")
    print(f"Features: {features}")
    return {
        "hypergraph": hyperedges,
        "features": features,
        "labels": {},
        "n": n,
    }


def compute_and_compare_encodings(hg1, hg2, name1="Graph1", name2="Graph2"):
    """Compute and compare encodings for two graphs.

    Args:
        hg1:
            the first hypergraph
        hg2:
            the second hypergraph
        name1:
            the name of the first graph
        name2:
            the name of the second graph
    """
    # Initialize encoder
    # not strictly necessary to have two encoders
    encoder_shrikhande = HypergraphEncodings()
    encoder_rooke = HypergraphEncodings()

    # Compute different encodings
    print(f"\nComputing encodings for {name1} and {name2}:")

    # Define encodings to check with their parameters
    encodings_to_check = [
        ("LDP", "Local Degree Profile", True),
        ("LCP-FRC", "Local Curvature Profile - FRC", True),
        ("RWPE", "Random Walk Encodings", True),
        ("LCP-ORC", "Local Curvature Profile - ORC", False),
    ]

    # Check each encoding type
    for encoding_type, description, should_be_same in encodings_to_check:
        print(f"\n=== {description} ===")
        checks_encodings(
            encoding_type,
            hg1,
            hg2,
            encoder_shrikhande,
            encoder_rooke,
            name1,
            name2,
        )

    # Laplacian encodings
    print("\n=== Laplacian Encodings ===")
    # Test all three Laplacian types
    for lap_type in ["Normalized", "RW", "Hodge"]:
        print(f"Laplacian type: {lap_type}")
        hg1_lape, hg2_lape, laplacian_1, laplacian_2, same = test_laplacian(
            hg1.copy(), hg2.copy(), lap_type, name1="Shrikhande", name2="Rooke"
        )


# Create and process both graphs
graphs = {
    "Shrikhande": nx.read_graph6("shrikhande.g6"),
    "Rooke": nx.read_graph6("rook_graph.g6"),
}

# Convert graphs to hypergraph dictionaries
hypergraphs = {name: convert_nx_to_hypergraph_dict(G) for name, G in graphs.items()}

# Now hypergraphs['Shrikhande'] and hypergraphs['Rooke'] contain the converted hypergraph dictionaries

# Print properties and generate plots for each graph
for name, G in graphs.items():
    print(f"\nProcessing {name} graph:")
    print_graph_properties(G)

    # Get and plot adjacency matrix as heatmap
    adj_matrix = nx.adjacency_matrix(G).todense()
    plt.figure(figsize=(8, 8))
    plt.imshow(adj_matrix, cmap="viridis")
    plt.colorbar()
    plt.title(f"Adjacency Matrix - {name} Graph")
    plt.savefig(f"{name.lower()}_adjacency_heatmap.png")
    plt.clf()

    # Draw the graph with circular layout
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True)
    plt.title(f"{name} Graph")
    plt.savefig(f"{name.lower()}_graph.png")
    plt.clf()

    # Plot degree distribution
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    plt.hist(degree_sequence, bins=range(1, max(degree_sequence) + 1))
    plt.title(f"Degree Distribution - {name} Graph")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.savefig(f"{name.lower()}_degree_distribution.png")
    plt.clf()

# Plot difference between adjacency matrices
plt.figure(figsize=(8, 8))
plt.imshow(
    np.abs(
        nx.adjacency_matrix(graphs["Shrikhande"]).todense()
        - nx.adjacency_matrix(graphs["Rooke"]).todense()
    ),
    cmap="Blues",
)
plt.colorbar()
plt.title("Difference in Adjacency Matrices (Shrikhande - Rooke)")
plt.savefig("adjacency_difference_heatmap.png")
plt.clf()


compute_and_compare_encodings(
    hypergraphs["Shrikhande"],
    hypergraphs["Rooke"],
    name1="Shrikhande",
    name2="Rooke",
)

# Then to do: use curvature to distinguish the two graphs.
# Then augement as hypergraph and show that LAPE can distinguish the two graphs.


# effect on U should just be swapping the rows
# TODO: do Forman curvature
