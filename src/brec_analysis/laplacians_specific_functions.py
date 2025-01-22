import os

import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
from torch_geometric.data import Data

from encodings_hnns.encodings import HypergraphEncodings
from encodings_hnns.laplacians import Laplacians


def compute_laplacian(hg, lap_type: str):
    """Compute Laplacian matrix for a given hypergraph."""
    encoder = HypergraphEncodings()

    # Initialize the encoder with hyperedges
    encoder.compute_hyperedges(hg, verbose=False)

    # Initialize and compute the Laplacian
    encoder.laplacian = Laplacians(hg)

    if lap_type == "Normalized":
        encoder.laplacian.compute_normalized_laplacian()
        L = encoder.laplacian.normalized_laplacian
    elif lap_type == "RW":
        encoder.laplacian.compute_random_walk_laplacian(verbose=False)
        L = encoder.laplacian.rw_laplacian
    elif lap_type == "Hodge":
        encoder.laplacian.compute_boundary()  # Need to compute boundary first
        encoder.laplacian.compute_hodge_laplacian()
        L = encoder.laplacian.hodge_laplacian_down

    hg_lape = encoder.add_laplacian_encodings(
        hg.copy(), type=lap_type, verbose=False, use_same_sign=True
    )
    del encoder
    return hg_lape, L


def reconstruct_matrix(eigenvalues, eigenvectors) -> np.ndarray:
    """Reconstruct the matrix from the eigenvalues and eigenvectors"""
    diagonal_matrix = np.diag(eigenvalues)
    reconstructed_matrix = eigenvectors @ diagonal_matrix @ eigenvectors.T
    return reconstructed_matrix


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
