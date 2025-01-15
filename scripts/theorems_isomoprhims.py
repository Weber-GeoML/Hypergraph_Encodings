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
"""

import json
import os
import re
import textwrap
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

import networkx as nx
from encodings_hnns.encodings import HypergraphEncodings


# Save matrices in pmatrix format
def matrix_to_pmatrix(matrix):
    latex_str = "\\begin{pmatrix}\n"
    for row in matrix:
        latex_str += " & ".join([f"{x:.4f}" for x in row]) + " \\\\\n"
    latex_str += "\\end{pmatrix}"
    return latex_str


def reconstruct_matrix(eigenvalues, eigenvectors):
    """Reconstruct the matrix from the eigenvalues and eigenvectors"""
    diagonal_matrix = np.diag(eigenvalues)
    reconstructed_matrix = eigenvectors @ diagonal_matrix @ eigenvectors.T
    return reconstructed_matrix


def test_laplacian(hg1, hg2, lap_type, name1="Graph1", name2="Graph2"):
    print(f"Testing Laplacian type: {lap_type}")
    # Initialize encoder
    encoder_shrikhande = HypergraphEncodings()
    encoder_rooke = HypergraphEncodings()

    hg1_lape = encoder_shrikhande.add_laplacian_encodings(
        hg1.copy(), type=lap_type, verbose=False
    )
    hg2_lape = encoder_rooke.add_laplacian_encodings(
        hg2.copy(), type=lap_type, verbose=False
    )

    # Get the appropriate Laplacian matrix based on type
    if lap_type == "Normalized":
        L1 = encoder_shrikhande.laplacian.normalized_laplacian
        L2 = encoder_rooke.laplacian.normalized_laplacian
    elif lap_type == "RW":
        L1 = encoder_shrikhande.laplacian.rw_laplacian
        L2 = encoder_rooke.laplacian.rw_laplacian
    else:  # Hodge
        L1 = encoder_shrikhande.laplacian.hodge_laplacian_up
        L2 = encoder_rooke.laplacian.hodge_laplacian_up
        L3 = encoder_shrikhande.laplacian.hodge_laplacian_down
        L4 = encoder_rooke.laplacian.hodge_laplacian_down

    # print the norms of the laplacians eigenvectors
    eigenvalues_shrikhande, eigenvectors_shrikhande = np.linalg.eigh(L1)
    eigenvalues_rooke, eigenvectors_rooke = np.linalg.eigh(L2)

    if lap_type == "Normalized" or lap_type == "Hodge":

        # assert that L1 and L2 are symmetric
        assert np.allclose(L1, L1.T, atol=1e-12, rtol=1e-12), "L1 is not symmetric"
        assert np.allclose(L2, L2.T, atol=1e-12, rtol=1e-12), "L2 is not symmetric"

        # print the min value in the eigenvectors
        print(f"Min value in eigenvectors of Shrikhande: {np.min(eigenvectors_shrikhande)}")
        print(f"Min value in eigenvectors of Rooke: {np.min(eigenvectors_rooke)}")
        # print the rank of the eigenvectors
        print(f"Rank of eigenvectors of Shrikhande: {np.linalg.matrix_rank(eigenvectors_shrikhande)}")
        print(f"Rank of eigenvectors of Rooke: {np.linalg.matrix_rank(eigenvectors_rooke)}")

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

        # Reconstructs the original matrix
        reconstructed_matrix = reconstruct_matrix(
            eigenvalues_shrikhande, eigenvectors_shrikhande
        )

        # Compare reconstructed matrix to the original matrix
        if np.allclose(reconstructed_matrix, L1, atol=1e-12, rtol=1e-12):
            print("Reconstructed matrix is close to the original matrix.")
            print("Symmetric matrix. Expected for Hodge, normalized")
        else:
            print("Reconstructed matrix differs from the original matrix.")
            print(
                f"Reconstructed matrix - original matrix: {reconstructed_matrix - L1}"
            )
            assert False

        # Reconstructs the original matrix
        reconstructed_matrix = reconstruct_matrix(eigenvalues_rooke, eigenvectors_rooke)

        # Compare reconstructed matrix to the original matrix
        if np.allclose(reconstructed_matrix, L2):
            print("Reconstructed matrix is close to the original matrix.")
            print("Symmetric matrix. Expected for Hodge, normalized")
        else:
            print("Reconstructed matrix differs from the original matrix.")
            assert False

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
        np.abs(hg1_lape["features"] - hg2_lape["features"]),
        cmap="hot",
        interpolation="nearest",
    )
    plt.colorbar()
    plt.savefig(f"diff_features_{lap_type.lower()}.png")
    plt.close()

    # save the diff of the laplacians as a matrix heatmap
    plt.imshow(np.abs(L1 - L2), cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.savefig(f"diff_laplacian_{lap_type.lower()}.png")
    plt.close()

    if lap_type == "Hodge":
        # save the diff of the laplacians as a matrix heatmap
        plt.imshow(np.abs(L3 - L4), cmap="hot", interpolation="nearest")
        plt.colorbar()
        plt.savefig(f"diff_laplacian_down_{lap_type.lower()}.png")
        plt.close()

    # print the laplacians
    if False:
        print(f"Shrikhande LAPE laplacian: {L1}")
        print(f"Rooke LAPE laplacian: {L2}")

    assert L1.shape == L2.shape, "The two laplacians have different shapes"

    # loop through row by row
    for i in range(L1.shape[0]):
        # print(f"Row {i} of Shrikhande LAPE laplacian: {L1[i]}")
        # print(f"Row {i} of Rooke LAPE laplacian: {L2[i]}")
        print(f"Diff of row {i}: {np.abs(L1[i] - L2[i])}")
    # loop through column by column
    for i in range(L1.shape[1]):
        print(f"Diff of column {i}: {np.abs(L1[:,i] - L2[:,i])}")

    with open(f"shrikhande_laplacian_{lap_type.lower()}.tex", "w") as f:
        f.write(matrix_to_pmatrix(L1))

    with open(f"rooke_laplacian_{lap_type.lower()}.tex", "w") as f:
        f.write(matrix_to_pmatrix(L2))

    # The Laplacians are different!
    if False:
        # print the max difference
        print(f"Max difference: {np.max(np.abs(L1 - L2))}")
        # print the difference matrix
        print(f"Difference matrix: {L1 - L2}")

        # assert that the two laplacians are the same
        assert np.allclose(
            L1, L2, rtol=1e-12, atol=1e-12
        ), "The two laplacians are not the same"

    # compute the eigenvalues
    eigenvalues_shrikhande, eigenvectors_shrikhande = np.linalg.eig(L1)
    eigenvalues_rooke, eigenvectors_rooke = np.linalg.eig(L2)
    are_isospectral = check_isospectrality(eigenvalues_shrikhande, eigenvalues_rooke)
    assert are_isospectral, "The two graphs are not isospectral"

    # TODO: we will do the same but at the hypegraph level.

    assert (
        hg1_lape["features"].shape == hg2_lape["features"].shape
    ), "The two LAPE encodings have different shapes"
    print("*" * 100)
    return hg1_lape, hg2_lape, L1, L2


def check_isospectrality(eig1, eig2, tolerance=1e-10, verbose=False):
    """
    Check if two graphs are isospectral by comparing their sorted eigenvalues.

    Args:
        eig1, eig2: Arrays of eigenvalues
        tolerance: Numerical tolerance for floating point comparison

    Returns:
        bool: True if graphs are isospectral
    """
    # Sort eigenvalues and take real parts
    eig1_sorted = np.sort(np.real(eig1))
    eig2_sorted = np.sort(np.real(eig2))

    # Check if arrays have same shape
    if eig1_sorted.shape != eig2_sorted.shape:
        return False

    # Compare eigenvalues within tolerance
    diff = np.abs(eig1_sorted - eig2_sorted)
    max_diff = np.max(diff)

    print(f"Maximum eigenvalue difference: {max_diff}")
    if verbose:
        print("\nSorted eigenvalues comparison:")
        for i, (e1, e2) in enumerate(zip(eig1_sorted, eig2_sorted)):
            print(f"λ{i+1}: {e1:.10f} vs {e2:.10f} (diff: {abs(e1-e2):.10f})")

    return max_diff < tolerance


def check_permutation_equivalence(matrix1, matrix2, tolerance=1e-15):
    """
    Check if two matrices can be made identical through row permutations.

    Args:
        matrix1, matrix2: numpy arrays of same shape
        tolerance: Numerical tolerance for floating point comparison

    Returns:
        bool: True if matrices can be made identical through permutation
        dict: Mapping of rows from matrix1 to matrix2 (if exists)
    """
    if matrix1.shape != matrix2.shape:
        print(
            f"The two matrices have different shapes: {matrix1.shape} and {matrix2.shape}"
        )
        return False, None

    # Find the permutation mapping
    mapping = {}
    used_indices = set()

    for i, row1 in enumerate(matrix1):
        print(f"Row {i} of matrix1: {row1}")
        found_match = False
        for j, row2 in enumerate(matrix2):
            if j not in used_indices and np.allclose(row1, row2, atol=tolerance):
                mapping[i] = j
                used_indices.add(j)
                found_match = True
                break
        if not found_match:
            return False, None

    return True, mapping


def print_graph_properties(G):
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


# Create a Shrikhande graph
G_shrikhande = nx.read_graph6("shrikhande.g6")
print_graph_properties(G_shrikhande)

# Get and plot adjacency matrix as heatmap
adj_matrix_shrikhande = nx.adjacency_matrix(G_shrikhande).todense()
plt.figure(figsize=(8, 8))
plt.imshow(adj_matrix_shrikhande, cmap="Blues")
plt.colorbar()
plt.title("Adjacency Matrix - Shrikhande Graph")
plt.savefig("shrikhande_adjacency_heatmap.png")
plt.clf()

# Draw the graph
# circular layout
pos = nx.circular_layout(G_shrikhande)
nx.draw(G_shrikhande, pos, with_labels=True)
# save the graph figure
plt.savefig("shrikhande_graph.png")
# clear the figure
plt.clf()
# print and save the degree distribution
degree_sequence = sorted([d for n, d in G_shrikhande.degree()], reverse=True)
plt.hist(degree_sequence, bins=range(1, max(degree_sequence) + 1))
plt.title("Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.savefig("shrikhande_degree_distribution.png")
# clear the figure
plt.clf()

# Do the same for Rooke graph
G_rooke = nx.read_graph6("rook_graph.g6")
print_graph_properties(G_rooke)

# Get and plot adjacency matrix as heatmap
adj_matrix_rooke = nx.adjacency_matrix(G_rooke).todense()
plt.figure(figsize=(8, 8))
plt.imshow(adj_matrix_rooke, cmap="Blues")
plt.colorbar()
plt.title("Adjacency Matrix - Rooke Graph")
plt.savefig("rooke_adjacency_heatmap.png")
plt.clf()

# Draw the graph
pos = nx.circular_layout(G_rooke)
nx.draw(G_rooke, pos, with_labels=True)
# Save the figure
plt.savefig("rooke_graph.png")
plt.clf()

# print and save the degree distribution
degree_sequence = sorted([d for n, d in G_rooke.degree()], reverse=True)
plt.hist(degree_sequence, bins=range(1, max(degree_sequence) + 1))
plt.title("Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.savefig("rooke_degree_distribution.png")
plt.clf()

# Also plot the difference between the adjacency matrices
plt.figure(figsize=(8, 8))
plt.imshow(np.abs(adj_matrix_shrikhande - adj_matrix_rooke), cmap="hot")
plt.colorbar()
plt.title("Difference in Adjacency Matrices")
plt.savefig("adjacency_difference_heatmap.png")
plt.clf()


# Using the relevant code, compute the encodings:


def convert_nx_to_hypergraph_dict(G):
    """Convert NetworkX graph to hypergraph dictionary format."""
    # Create edge dictionary where each edge is a hyperedge of size 2
    hyperedges = {f"e_{i}": list(edge) for i, edge in enumerate(G.edges())}

    # Create features matrix (initially empty)
    n = G.number_of_nodes()
    features = np.empty((n, 0))

    print(f"Hypergraph: {hyperedges}")
    print(f"Features: {features}")
    return {"hypergraph": hyperedges, "features": features, "labels": {}, "n": n}


def compute_and_compare_encodings(G1, G2, name1="Graph1", name2="Graph2"):
    """Compute and compare encodings for two graphs."""
    # Convert graphs to hypergraph format
    hg1 = convert_nx_to_hypergraph_dict(G1)
    hg2 = convert_nx_to_hypergraph_dict(G2)

    # Initialize encoder
    # not strictly necessary to have two encoders
    encoder_shrikhande = HypergraphEncodings()
    encoder_rooke = HypergraphEncodings()

    # Compute different encodings
    print(f"\nComputing encodings for {name1} and {name2}:")

    # Local degree profile
    print("\n=== Local Degree Profile ===")
    hg1_ldp = encoder_shrikhande.add_degree_encodings(hg1.copy(), verbose=False)
    hg2_ldp = encoder_rooke.add_degree_encodings(hg2.copy(), verbose=False)
    # assert that the two degree distributions are the same
    assert np.all(
        hg1_ldp["features"] == hg2_ldp["features"]
    ), "The two degree distributions are not the same"
    print(f"The two degree distributions are the same!")
    print(f"\n{name1} LDP shape:", hg1_ldp["features"].shape)
    print(f"{name2} LDP shape:", hg2_ldp["features"].shape)

    # Count unique features (element-wise)
    # Count unique rows (node-wise)
    unique_rows_1 = np.unique(hg1_ldp["features"], axis=0)
    unique_rows_2 = np.unique(hg2_ldp["features"], axis=0)
    print(f"\nUnique node feature vectors in {name1}:")
    print(f"Number of unique rows: {unique_rows_1.shape[0]}")
    print("Values:")
    for i, row in enumerate(unique_rows_1):
        print(f"Pattern {i+1}: {row}")
        # Count how many nodes have this pattern
        count = np.sum(np.all(hg1_ldp["features"] == row, axis=1))
        print(f"Frequency: {count} nodes")

    print(f"\nUnique node feature vectors in {name2}:")
    print(f"Number of unique rows: {unique_rows_2.shape[0]}")
    print("Values:")
    for i, row in enumerate(unique_rows_2):
        print(f"Pattern {i+1}: {row}")
        # Count how many nodes have this pattern
        count = np.sum(np.all(hg2_ldp["features"] == row, axis=1))
        print(f"Frequency: {count} nodes")

    # print the encodings
    # print(f"Shrikhande LDP encodings: {hg1_ldp['features']}")
    # print(f"Rooke LDP encodings: {hg2_ldp['features']}")

    # Random Walk encodings
    print("\n=== Random Walk Encodings ===")
    # Initialize encoder
    encoder_shrikhande = HypergraphEncodings()
    encoder_rooke = HypergraphEncodings()
    hg1_rw = encoder_shrikhande.add_randowm_walks_encodings(
        hg1.copy(), rw_type="WE", verbose=False
    )
    hg2_rw = encoder_rooke.add_randowm_walks_encodings(
        hg2.copy(), rw_type="WE", verbose=False
    )
    print(
        f"Maximum difference: {np.max(np.abs(hg1_rw['features'] - hg2_rw['features']))}"
    )

    # assert that the two random walk distributions are the same
    assert np.allclose(
        hg1_rw["features"], hg2_rw["features"], rtol=1e-12, atol=1e-12
    ), "The two random walk distributions differ beyond 1e-12 tolerance"
    print(f"The two random walk distributions are the same!")
    print(f"\n{name1} RW shape:", hg1_rw["features"].shape)
    print(f"{name2} RW shape:", hg2_rw["features"].shape)

    # print the encodings
    if False:
        print(f"Shrikhande RW encodings: {hg1_rw['features']}")
        print(f"Rooke RW encodings: {hg2_rw['features']}")

    # Laplacian encodings
    print("\n=== Laplacian Encodings ===")
    # Test all three Laplacian types
    for lap_type in ["Normalized", "RW", "Hodge"]:
        print(f"Laplacian type: {lap_type}")
        hg1_lape, hg2_lape, L1, L2 = test_laplacian(
            hg1.copy(), hg2.copy(), lap_type, name1="Shrikhande", name2="Rooke"
        )

    return {
        "lape": (hg1_lape["features"], hg2_lape["features"]),
        "rw": (hg1_rw["features"], hg2_rw["features"]),
    }


hg_shrikhande = convert_nx_to_hypergraph_dict(G_shrikhande)
hg_rooke = convert_nx_to_hypergraph_dict(G_rooke)
compute_and_compare_encodings(G_shrikhande, G_rooke, name1="Shrikhande", name2="Rooke")

# Then to do: use curvature to distinguish the two graphs.
# Then augement as hypergraph and show that LAPE can distinguish the two graphs.
