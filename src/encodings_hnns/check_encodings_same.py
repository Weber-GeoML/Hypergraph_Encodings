
import itertools
import json
import os
import re
import textwrap

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from tabulate import tabulate

from encodings_hnns.encodings import HypergraphEncodings


def checks_encodings(
    name_of_encoding: str,
    same: bool,
    hg1,
    hg2,
    encoder_shrikhande,
    encoder_rooke,
    name1: str = "Shrikhande",
    name2: str = "Rooke",
) -> None:
    # Initialize encoder
    encoder_shrikhande = HypergraphEncodings()
    encoder_rooke = HypergraphEncodings()
    print(f"\n=== {name_of_encoding} ===")
    if name_of_encoding == "LDP":
        hg1_encodings = encoder_shrikhande.add_degree_encodings(
            hg1.copy(), verbose=False
        )
        hg2_encodings = encoder_rooke.add_degree_encodings(hg2.copy(), verbose=False)
    elif name_of_encoding == "LAPE":
        hg1_encodings = encoder_shrikhande.add_laplacian_encodings(
            hg1.copy(), type="Normalized", verbose=False
        )
        hg2_encodings = encoder_rooke.add_laplacian_encodings(
            hg2.copy(), type="Normalized", verbose=False
        )
    elif name_of_encoding == "RWPE":
        hg1_encodings = encoder_shrikhande.add_randowm_walks_encodings(
            hg1.copy(), rw_type="WE", verbose=False
        )
        hg2_encodings = encoder_rooke.add_randowm_walks_encodings(
            hg2.copy(), rw_type="WE", verbose=False
        )
    elif name_of_encoding == "LCP-ORC":
        hg1_encodings = encoder_shrikhande.add_curvature_encodings(
            hg1.copy(), verbose=False, type="ORC"
        )
        hg2_encodings = encoder_rooke.add_curvature_encodings(
            hg2.copy(), verbose=False, type="ORC"
        )
    elif name_of_encoding == "LCP-FRC":
        hg1_encodings = encoder_shrikhande.add_curvature_encodings(
            hg1.copy(), verbose=False, type="FRC"
        )
        hg2_encodings = encoder_rooke.add_curvature_encodings(
            hg2.copy(), verbose=False, type="FRC"
        )
    # assert that the two degree distributions are the same
    if same:
        rtol = 1e-10  # relative tolerance
        atol = 1e-10  # absolute tolerance
        diffs = np.abs(hg1_encodings["features"] - hg2_encodings["features"])
        assert np.allclose(
            hg1_encodings["features"],
            hg2_encodings["features"],
            rtol=rtol,
            atol=atol
        ), f"The two distributions differ beyond tolerance. Max difference: {np.max(diffs)}. \n Difference matrix: {diffs}"
    print(f"The two degree distributions are the same!")
    print(f"\n{name1} {name_of_encoding} shape:", hg1_encodings["features"].shape)
    print(f"{name2} {name_of_encoding} shape:", hg2_encodings["features"].shape)

    # Count unique features (element-wise)
    # Count unique rows (node-wise)
    unique_rows_1 = np.unique(hg1_encodings["features"], axis=0)
    unique_rows_2 = np.unique(hg2_encodings["features"], axis=0)
    print(f"\nUnique node feature vectors in {name1}:")
    print(f"Number of unique rows: {unique_rows_1.shape[0]}")
    print("Values:")
    for i, row in enumerate(unique_rows_1):
        print(f"Pattern {i+1}: {row}")
        # Count how many nodes have this pattern
        count = np.sum(np.all(hg1_encodings["features"] == row, axis=1))
        print(f"Frequency: {count} nodes")

    print(f"\nUnique node feature vectors in {name2}:")
    print(f"Number of unique rows: {unique_rows_2.shape[0]}")
    print("Values:")
    for i, row in enumerate(unique_rows_2):
        print(f"Pattern {i+1}: {row}")
        # Count how many nodes have this pattern
        count = np.sum(np.all(hg2_encodings["features"] == row, axis=1))
        print(f"Frequency: {count} nodes")


# Save matrices in pmatrix format
def matrix_to_pmatrix(matrix) -> str:
    latex_str = "\\begin{pmatrix}\n"
    for row in matrix:
        latex_str += " & ".join([f"{x:.4f}" for x in row]) + " \\\\\n"
    latex_str += "\\end{pmatrix}"
    return latex_str


def reconstruct_matrix(eigenvalues, eigenvectors) -> np.ndarray:
    """Reconstruct the matrix from the eigenvalues and eigenvectors"""
    diagonal_matrix = np.diag(eigenvalues)
    reconstructed_matrix = eigenvectors @ diagonal_matrix @ eigenvectors.T
    return reconstructed_matrix


def test_laplacian(hg1, hg2, lap_type, name1 : str = "Graph1", name2 : str = "Graph2", verbose : bool = False):
    """
    Args:
        hg1, hg2: hypergraphs
        lap_type: type of Laplacian to use
        name1, name2: names of the graphs
        verbose: whether to print verbose output
    Returns:
        hg1_lape, hg2_lape: LAPE encodings of the two graphs
        L1, L2: Laplacian matrices of the two graphs
        same: whether the two graphs are the same
    TODO: to finish! Should be easy and fast now.
    """
    if verbose:
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
        Dv1 = encoder_shrikhande.laplacian.Dv
        Dv2 = encoder_rooke.laplacian.Dv
        assert np.allclose(
            Dv1, 6 * np.eye(Dv1.shape[0]), atol=1e-12, rtol=1e-12
        ), "Dv is not the identity matrix"
        assert np.allclose(
            Dv2, 6 * np.eye(Dv2.shape[0]), atol=1e-12, rtol=1e-12
        ), "Dv is not the identity matrix"
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

    # print the eigenvalues
    # print(f"Eigenvalues of Shrikhande Laplacian: {eigenvalues_shrikhande}")
    # print(f"Eigenvalues of Rooke Laplacian: {eigenvalues_rooke}")
    # assert they are in order. Smaller to bigger.
    assert np.allclose(eigenvalues_shrikhande, np.sort(eigenvalues_shrikhande)), "Eigenvalues of Shrikhande Laplacian are not in order"
    assert np.allclose(eigenvalues_rooke, np.sort(eigenvalues_rooke)), "Eigenvalues of Rooke Laplacian are not in order"

    if lap_type == "Normalized" or lap_type == "Hodge":

        # assert that L1 and L2 are symmetric
        assert np.allclose(L1, L1.T, atol=1e-12, rtol=1e-12), "L1 is not symmetric"
        assert np.allclose(L2, L2.T, atol=1e-12, rtol=1e-12), "L2 is not symmetric"

        # Can already the min value, max value and other statistics
        # to check wether the encodings are the same or not!!!!
        # TODO!
        min_shrikhande = np.min(eigenvectors_shrikhande)
        min_rooke = np.min(eigenvectors_rooke)
        if verbose:
            # print the min value in the eigenvectors
            print(
                f"Min value in eigenvectors of Shrikhande: {min_shrikhande}"
            )
            print(f"Min value in eigenvectors of Rooke: {min_rooke}")
        # ranks
        rank_shrikhande = np.linalg.matrix_rank(eigenvectors_shrikhande)
        rank_rooke = np.linalg.matrix_rank(eigenvectors_rooke)
        # print the rank of the eigenvectors
        print(
            f"Rank of eigenvectors of Shrikhande: {rank_shrikhande}"
        )
        print(
            f"Rank of eigenvectors of Rooke: {rank_rooke}"
        )
        if rank_shrikhande != rank_rooke:
            print("The two graphs have different encodings")
            return hg1_lape, hg2_lape, L1, L2, False
        if min_shrikhande != min_rooke:
            print("The two graphs have different encodings")
            return hg1_lape, hg2_lape, L1, L2, False


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
            ('Shrikhande', L1, eigenvalues_shrikhande, eigenvectors_shrikhande),
            ('Rooke', L2, eigenvalues_rooke, eigenvectors_rooke)
        ]

        # Store properties for comparison
        properties = {name: {} for name, *_ in matrices_to_check}

        # Check all properties for each matrix
        for name, original_matrix, eigenvalues, eigenvectors in matrices_to_check:
            # Check symmetry
            assert np.allclose(original_matrix, original_matrix.T, atol=1e-12, rtol=1e-12), \
                f"{name} matrix is not symmetric"

            # Store basic properties
            properties[name].update({
                'min_eigenvector': np.min(eigenvectors),
                'rank': np.linalg.matrix_rank(eigenvectors),
                'norms': np.sort(np.linalg.norm(eigenvectors, axis=1))
            })

            if verbose:
                print(f"\nProperties for {name}:")
                print(f"Min value in eigenvectors: {properties[name]['min_eigenvector']}")
                print(f"Rank of eigenvectors: {properties[name]['rank']}")
                print(f"Sorted norms of eigenvectors: {properties[name]['norms']}")

            # Check orthonormality
            orthonormal_diff = eigenvectors @ eigenvectors.T - np.eye(eigenvectors.shape[0])
            assert np.allclose(
                eigenvectors @ eigenvectors.T,
                np.eye(eigenvectors.shape[0]),
                atol=1e-12,
                rtol=1e-12
            ), f"Eigenvectors of {name} are not orthonormal. Difference: {orthonormal_diff}"

            # Check matrix reconstruction
            reconstructed_matrix = reconstruct_matrix(eigenvalues, eigenvectors)
            if np.allclose(reconstructed_matrix, original_matrix, atol=1e-12, rtol=1e-12):
                print(f"{name} reconstructed matrix is close to the original matrix.")
                print("Symmetric matrix. Expected for Hodge, normalized")
            else:
                print(f"{name} reconstructed matrix differs from the original matrix.")
                print(f"Difference matrix:\n{reconstructed_matrix - original_matrix}")
                max_diff = np.max(np.abs(reconstructed_matrix - original_matrix))
                print(f"Maximum difference: {max_diff}")
                assert False, f"{name} matrix reconstruction failed"

        # Compare properties between graphs
        for prop in ['min_eigenvector', 'rank']:
            if properties['Shrikhande'][prop] != properties['Rooke'][prop]:
                print(f"The two graphs have different encodings (different {prop})")
                return hg1_lape, hg2_lape, L1, L2, False

        # Compare norms
        if not np.allclose(properties['Shrikhande']['norms'], 
                          properties['Rooke']['norms'], 
                          atol=1e-12, rtol=1e-12):
            print("The two graphs have different encodings (different eigenvector norms)")
            return hg1_lape, hg2_lape, L1, L2, False

        print("\nComparison of eigenvector norms:")
        for name in ['Shrikhande', 'Rooke']:
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
    plt.imshow(L1 - L2, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title(f"Difference in {lap_type} Laplacian Matrices")
    plt.savefig(f"diff_laplacian_{lap_type.lower()}.png")
    plt.close()

    if lap_type == "Hodge":
        # save the diff of the laplacians as a matrix heatmap
        plt.imshow(L3 - L4, cmap="hot", interpolation="nearest")
        plt.colorbar()
        plt.title(f"Difference in {lap_type} Down-Laplacian Matrices")
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
    return hg1_lape, hg2_lape, L1, L2, True


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


# def check_permutation_equivalence(matrix1, matrix2, tolerance=1e-15):
#     """
#     Check if two matrices can be made identical through row permutations.

#     Args:
#         matrix1, matrix2: numpy arrays of same shape
#         tolerance: Numerical tolerance for floating point comparison

#     Returns:
#         bool: True if matrices can be made identical through permutation
#         dict: Mapping of rows from matrix1 to matrix2 (if exists)
#     """
#     if matrix1.shape != matrix2.shape:
#         print(
#             f"The two matrices have different shapes: {matrix1.shape} and {matrix2.shape}"
#         )
#         return False, None

#     # Find the permutation mapping
#     mapping = {}
#     used_indices = set()

#     for i, row1 in enumerate(matrix1):
#         print(f"Row {i} of matrix1: {row1}")
#         found_match = False
#         for j, row2 in enumerate(matrix2):
#             if j not in used_indices and np.allclose(row1, row2, atol=tolerance):
#                 mapping[i] = j
#                 used_indices.add(j)
#                 found_match = True
#                 break
#         if not found_match:
#             return False, None

#     return True, mapping