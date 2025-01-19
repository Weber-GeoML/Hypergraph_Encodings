import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import permutations
from encodings_hnns.encodings import HypergraphEncodings
from encodings_hnns.laplacians import Laplacians


def find_encoding_match(encoding1, encoding2):
    """
    Check if two encodings are equivalent under row permutations.
    Returns (is_match, permuted_encoding1, permutation) if found, (False, None, None) if not.
    
    Args:
        encoding1: numpy array of shape (n, d)
        encoding2: numpy array of shape (n, d)
    
    Returns:
        is_match: whether the two encodings are the same
        permuted_encoding1: the permuted encoding of encoding1
        permutation: the permutation that was applied
    """
    if encoding1.shape != encoding2.shape:
        return False, None, None
    
    # First check if the encodings are identical
    if np.allclose(encoding1, encoding2, rtol=1e-10):
        # Return identity permutation if encodings are identical
        print("Free lunch!")
        n_rows = encoding1.shape[0]
        return True, encoding1, tuple(range(n_rows))
    
    # Check the max absolute value of each encodings. If they are different, return False
    if not np.isclose(np.max(np.abs(encoding1)), np.max(np.abs(encoding2)), rtol=1e-3):
        print(f"Max absolute value of encoding1: {np.max(np.abs(encoding1))}")
        print(f"Max absolute value of encoding2: {np.max(np.abs(encoding2))}")
        return False, None, None

    # Same for min
    if not np.isclose(np.min(np.abs(encoding1)), np.min(np.abs(encoding2)), rtol=1e-3):
        print(f"Min absolute value of encoding1: {np.min(np.abs(encoding1))}")
        print(f"Min absolute value of encoding2: {np.min(np.abs(encoding2))}")
        return False, None, None
    
    # Compare the last column only. IF the max absolute value of the last column is different, return False
    if not np.isclose(np.max(np.abs(encoding1[:, -1])), np.max(np.abs(encoding2[:, -1])), rtol=1e-3):
        print(f"Max absolute value of last column of encoding1: {np.max(np.abs(encoding1[:, -1]))}")
        print(f"Max absolute value of last column of encoding2: {np.max(np.abs(encoding2[:, -1]))}")
        return False, None, None

    # Compare the first column only. If the max absolute value of the first column is different, return False
    if not np.isclose(np.max(np.abs(encoding1[:, 0])), np.max(np.abs(encoding2[:, 0])), rtol=1e-3):
        print(f"Max absolute value of first column of encoding1: {np.max(np.abs(encoding1[:, 0]))}")
        print(f"Max absolute value of first column of encoding2: {np.max(np.abs(encoding2[:, 0]))}")
        return False, None, None
    
    n_rows = encoding1.shape[0]
    
    # For small matrices, we can try all permutations
    if n_rows <= 10:  # Adjust this threshold based on your needs
        for perm in permutations(range(n_rows)):
            permuted = encoding1[list(perm), :]
            if np.allclose(permuted, encoding2, rtol=1e-10):
                return True, permuted, perm
    else:
        # For larger matrices, use a heuristic approach
        # Sort rows lexicographically and compare
        sorted1 = encoding1[np.lexsort(encoding1.T)]
        sorted2 = encoding2[np.lexsort(encoding2.T)]
        if np.allclose(sorted1, sorted2):
            # Find the permutation that was applied
            perm = np.argsort(np.lexsort(encoding1.T))
            return True, sorted1, perm
    
    return False, None, None


def plot_matched_encodings(encoding1, encoding2, ax1, ax2, ax3, name1="Graph A", name2="Graph B", title=""):
    """
    Plot two encodings and their difference, attempting to match their row orderings if possible.
    
    Args:
        encoding1, encoding2: numpy arrays of shape (n, d)
        ax1, ax2, ax3: matplotlib axes for plotting
        name1, name2: names of the graphs
        title: title for the plots
    
    Returns:
        is_match: bool indicating if a matching permutation was found
        permuted: permuted version of encoding1 if match found, None otherwise
        perm: permutation used if match found, None otherwise
    """
    is_match, permuted, perm = find_encoding_match(encoding1, encoding2)
    
    if is_match:
        im1 = ax1.imshow(permuted, cmap="viridis")
        im2 = ax2.imshow(encoding2, cmap="viridis")
        diff = np.abs(permuted - encoding2)
        # add the min and max value of the encoding to the title
        ax1.set_title(f"{name1}\n(Permuted to match {name2}) \n min: {np.min(permuted):.2e}, max: {np.max(permuted):.2e}")
    else:
        im1 = ax1.imshow(encoding1, cmap="viridis")
        im2 = ax2.imshow(encoding2, cmap="viridis")
        diff = encoding1 - encoding2
        # add the min and max value of the encoding to the title
        ax1.set_title(f"{name1}\n(Original ordering) \n min: {np.min(encoding1):.2e}, max: {np.max(encoding1):.2e}")
    
    # Plot difference matrix
    im3 = ax3.imshow(diff, cmap="Blues")  # Using Reds colormap to highlight differences
    
    # add the min and max value of the encoding two to the title
    ax2.set_title(f"{name2}\n(min: {np.min(encoding2):.2e}, max: {np.max(encoding2):.2e})")
    
    # Check if difference is uniformly zero
    if np.allclose(diff, np.zeros_like(diff)):
        ax3.set_title("Absolute Difference\n(Uniformly Zero)")
    else:
        # Get max absolute values for both encodings
        max_abs1 = np.max(np.abs(encoding1))
        max_abs2 = np.max(np.abs(encoding2))
        ax3.set_title(f"Difference\nMax abs values: {max_abs1:.2e} vs {max_abs2:.2e}\n Mean abs values: {np.mean(np.abs(encoding1)):.2e} vs {np.mean(np.abs(encoding2)):.2e}")
    
    # Add colorbars
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    plt.colorbar(im3, ax=ax3)
    
    # Add row labels if the matrices are small enough
    if encoding1.shape[0] <= 10:
        # For the first plot, use permuted node ordering if a match was found
        if is_match:
            for i, p in enumerate(perm):
                ax1.text(-0.5, i, f"Node {p}", va='center')
        else:
            for i in range(encoding1.shape[0]):
                ax1.text(-0.5, i, f"Node {i}", va='center')
        
        # For second plot, always use original ordering
        for i in range(encoding2.shape[0]):
            ax2.text(-0.5, i, f"Node {i}", va='center')
            ax3.text(-0.5, i, f"Node {i}", va='center')
    
    return is_match, permuted, perm

def checks_encodings(
    name_of_encoding: str,
    same: bool,
    hg1,
    hg2,
    encoder_shrikhande,
    encoder_rooke,
    name1: str = "Graph A",
    name2: str = "Graph B",
    save_plots: bool = True,
    plot_dir: str = "plots/encodings",
    pair_idx: int = None,
    category: str = None,
    is_isomorphic: bool = None,
    node_mapping: dict = None,
) -> bool:
    """Check if two graphs have the same encodings."""
    # Handle Laplacian encodings separately
    if name_of_encoding.startswith("LAPE-"):
        lap_type = name_of_encoding.split("-")[1]  # Get Normalized, RW, or Hodge
        # Get Laplacian matrices and features
        hg1_lape, L1 = compute_laplacian(hg1, lap_type)
        hg2_lape, L2 = compute_laplacian(hg2, lap_type)
        
        # Compute eigendecomposition
        eigenvalues1, eigenvectors1 = np.linalg.eigh(L1)  # Using eigh for symmetric matrices
        eigenvalues2, eigenvectors2 = np.linalg.eigh(L2)
        
        # Verify eigenvalues are sorted
        assert np.allclose(eigenvalues1, np.sort(eigenvalues1)), "Eigenvalues of Graph A are not in order"
        assert np.allclose(eigenvalues2, np.sort(eigenvalues2)), "Eigenvalues of Graph B are not in order"
        
        # Check matrix properties
        properties = {
            'Graph A': {
                'min_eigenvector': np.min(eigenvectors1),
                'rank': np.linalg.matrix_rank(eigenvectors1),
                'norms': np.sort(np.linalg.norm(eigenvectors1, axis=1))
            },
            'Graph B': {
                'min_eigenvector': np.min(eigenvectors2),
                'rank': np.linalg.matrix_rank(eigenvectors2),
                'norms': np.sort(np.linalg.norm(eigenvectors2, axis=1))
            }
        }
        
        # Initialize result flags
        same_properties = True
        
        # Compare properties
        for prop in ['min_eigenvector', 'rank']:
            if not np.allclose(properties['Graph A'][prop], properties['Graph B'][prop], rtol=1e-10):
                print(f"The two graphs have different {prop} for {name_of_encoding}")
                same_properties = False
        
        # Compare norms
        same_norms = np.allclose(properties['Graph A']['norms'], 
                               properties['Graph B']['norms'], 
                               atol=1e-12, rtol=1e-12)
        if not same_norms:
            print(f"The two graphs have different eigenvector norms for {name_of_encoding}")
            same_properties = False
            
        # Print comparison of norms
        print("\nComparison of eigenvector norms:")
        for name in ['Graph A', 'Graph B']:
            print(f"{name} Laplacian eigenvector norms: {properties[name]['norms']}")
        
        # Create figure and subplots
        plt.figure(figsize=(18, 5))
        title = f"{name_of_encoding} Comparison"
        if pair_idx is not None and category is not None:
            title += f"\nPair {pair_idx} ({category})"
        plt.suptitle(title, fontsize=14, y=1.05)
        
        # Create three subplots
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)
        
        # Try to find matching permutation for eigenvectors
        is_match, permuted, perm = plot_matched_encodings(
            eigenvectors1,
            eigenvectors2,
            ax1,
            ax2,
            ax3,
            name1,
            name2,
            title
        )
        
        # Print results and save plot
        print_comparison_results(is_match, name_of_encoding, perm, permuted, 
                               {"features": eigenvectors1}, {"features": eigenvectors2})
        
        if save_plots:
            save_comparison_plot(plt, plot_dir, pair_idx, category, name_of_encoding)
            
            # Save additional plots
            # Features difference plot
            plt.figure()
            plt.imshow(hg1_lape["features"] - hg2_lape["features"], cmap="Blues")
            plt.colorbar()
            plt.title(f"Difference in {lap_type} Features\n{category} - Pair {pair_idx}")
            save_comparison_plot(plt, plot_dir, pair_idx, category, f"{name_of_encoding}_features")
            plt.close()
            
            # Laplacian matrices difference plot
            plt.figure()
            plt.imshow(L1 - L2, cmap="Blues")
            plt.colorbar()
            plt.title(f"Difference in {lap_type} Laplacian Matrices\n{category} - Pair {pair_idx}")
            save_comparison_plot(plt, plot_dir, pair_idx, category, f"{name_of_encoding}_matrices")
            plt.close()
            
        
        plt.close()
        
        # Check isospectrality
        are_isospectral = check_isospectrality(eigenvalues1, eigenvalues2)
        if not are_isospectral:
            print(f"\n❌ The two graphs are not isospectral for {name_of_encoding}")
        else:
            print(f"\n✅ The two graphs are isospectral for {name_of_encoding}")
        # print is same properties with box
        print(f"\n{'✅' if same_properties else '❌'} Properties comparison for {name_of_encoding}")
        
        final_result = is_match
        print(f"\n{'✅✅✅' if final_result else '❌❌❌'} {name_of_encoding} comparison result")
        return final_result
    
    # Original code for other encodings
    hg1_encodings = get_encodings(hg1, encoder_shrikhande, name_of_encoding)
    hg2_encodings = get_encodings(hg2, encoder_rooke, name_of_encoding)
    
    # Create figure and subplots with adjusted width for three plots
    plt.figure(figsize=(18, 5))  # Increased width to accommodate third subplot
    title = f"{name_of_encoding} Encodings Comparison"
    if pair_idx is not None and category is not None:
        title += f"\nPair {pair_idx} ({category})"
    plt.suptitle(title, fontsize=14, y=1.05)
    
    # Create three subplots
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
    
    # Plot encodings using the shared function
    is_match, permuted, perm = plot_matched_encodings(
        hg1_encodings["features"],
        hg2_encodings["features"],
        ax1,
        ax2,
        ax3,
        name1,
        name2,
        title
    )
    
    # Print results and save plot
    print_comparison_results(is_match, name_of_encoding, perm, permuted, hg1_encodings, hg2_encodings)
    
    if save_plots:
        save_comparison_plot(plt, plot_dir, pair_idx, category, name_of_encoding)
    
    plt.close()
    return is_match


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





def find_isomorphism_mapping(G1, G2):
    """
    Find the node mapping between two isomorphic graphs with detailed debugging.
    """
    import networkx.algorithms.isomorphism as iso
    import networkx as nx
    
    # Convert graphs to simple undirected graphs
    G1 = nx.Graph(G1)
    G2 = nx.Graph(G2)
    
    print("\n=== Detailed Isomorphism Check ===")
    print("\nGraph Properties:")
    print(f"G1: {len(G1)} nodes, {G1.number_of_edges()} edges")
    print(f"G2: {len(G2)} nodes, {G2.number_of_edges()} edges")
    
    print("\nNode Degrees:")
    print("G1 degrees:", sorted([d for n, d in G1.degree()]))
    print("G2 degrees:", sorted([d for n, d in G2.degree()]))
    
    print("\nEdge Lists:")
    print("G1 edges:", sorted(G1.edges()))
    print("G2 edges:", sorted(G2.edges()))
    
    class VerboseGraphMatcher(iso.GraphMatcher):
        def __init__(self, G1, G2):
            super().__init__(G1, G2)
            self.mapping_steps = []
        
        def semantic_feasibility(self, G1_node, G2_node):
            """Print detailed information about node matching attempts"""
            feasible = super().semantic_feasibility(G1_node, G2_node)
            print(f"\nTrying to match:")
            print(f"G1 node {G1_node} (degree {G1.degree[G1_node]}) with")
            print(f"G2 node {G2_node} (degree {G2.degree[G2_node]})")
            print(f"Current mapping: {self.mapping}")
            print(f"Feasible: {feasible}")
            if not feasible:
                print("Neighbors comparison:")
                print(f"G1 node {G1_node} neighbors: {list(G1.neighbors(G1_node))}")
                print(f"G2 node {G2_node} neighbors: {list(G2.neighbors(G2_node))}")
            return feasible
    
    try:
        # Create verbose graph matcher
        GM = VerboseGraphMatcher(G1, G2)
        
        # Check isomorphism
        is_isomorphic = GM.is_isomorphic()
        
        if is_isomorphic:
            mapping = GM.mapping
            print("\n✅ Graphs are isomorphic!")
            print(f"Final mapping: {mapping}")
            
            # Verify mapping
            print("\nVerifying mapping...")
            for edge in G1.edges():
                mapped_edge = (mapping[edge[0]], mapping[edge[1]])
                if not G2.has_edge(*mapped_edge):
                    print(f"❌ Mapping verification failed for edge {edge}!")
                    return None
                print(f"✓ Edge {edge} correctly maps to {mapped_edge}")
            
            return mapping
        else:
            print("\n❌ Graphs are not isomorphic")
            print("Possible reasons:")
            print("1. Different degree sequences")
            print("2. Different neighborhood structures")
            print("3. No valid node mapping preserves all edges")
            return None
            
    except Exception as e:
        print(f"\n❌ Error during isomorphism check: {str(e)}")
        return None

def get_encodings(hg, encoder, name_of_encoding):
    """Helper function to get the appropriate encodings based on type."""
    if name_of_encoding == "LDP":
        return encoder.add_degree_encodings(hg.copy(), verbose=False)
    elif name_of_encoding == "LAPE":
        return encoder.add_laplacian_encodings(hg.copy(), type="Normalized", verbose=False)
    elif name_of_encoding == "RWPE":
        return encoder.add_randowm_walks_encodings(hg.copy(), rw_type="WE", verbose=False)
    elif name_of_encoding == "LCP-ORC":
        return encoder.add_curvature_encodings(hg.copy(), verbose=False, type="ORC")
    elif name_of_encoding == "LCP-FRC":
        return encoder.add_curvature_encodings(hg.copy(), verbose=False, type="FRC")
    elif name_of_encoding == "LAPE-Normalized":
        return encoder.add_laplacian_encodings(hg.copy(), type="Normalized", verbose=False)
    elif name_of_encoding == "LAPE-RW":
        return encoder.add_laplacian_encodings(hg.copy(), type="RW", verbose=False)
    elif name_of_encoding == "LAPE-Hodge":
        return encoder.add_laplacian_encodings(hg.copy(), type="Hodge", verbose=False)

def print_comparison_results(is_match, name_of_encoding, perm, permuted, hg1_encodings, hg2_encodings):
    """Helper function to print comparison results."""
    # Check if we're dealing with Laplacian encodings
    is_laplacian = name_of_encoding.startswith("LAPE-")
    
    if is_match:
        print(f"\n✅ Found matching permutation for {name_of_encoding}!")
        print(f"Permutation: {perm}")
        print("Statistics after permutation:")
        print(f"Max difference: {np.max(np.abs(permuted - hg2_encodings['features']))}")
        print(f"Mean difference: {np.mean(np.abs(permuted - hg2_encodings['features']))}")
        if not is_laplacian:  # Only print extra newline for non-Laplacian encodings
            print(f"\n")
    else:
        print(f"\n❌ No matching permutation found for {name_of_encoding}")
        print("Differences in original ordering:")
        diff = np.abs(hg1_encodings['features'] - hg2_encodings['features'])
        print(f"Max difference: {np.max(diff)}")
        print(f"Mean difference: {np.mean(diff)}")
        print(f"Max abs values: {np.max(np.abs(hg1_encodings['features']))} vs {np.max(np.abs(hg2_encodings['features']))}")
        print(f"Min abs values: {np.min(np.abs(hg1_encodings['features']))} vs {np.min(np.abs(hg2_encodings['features']))}")
        print(f"Mean abs values: {np.mean(np.abs(hg1_encodings['features']))} vs {np.mean(np.abs(hg2_encodings['features']))}")
        if not is_laplacian:  # Only print extra newline for non-Laplacian encodings
            print(f"\n")

def save_comparison_plot(plt, plot_dir, pair_idx, category, name_of_encoding):
    """Helper function to save the comparison plot."""
    os.makedirs(plot_dir, exist_ok=True)
    filename_base = f"pair_{pair_idx}_{category.lower()}" if pair_idx is not None else "comparison"
    plt.savefig(
        f"{plot_dir}/{filename_base}_{name_of_encoding.lower()}_comparison.png",
        bbox_inches='tight',
        dpi=300
    )

def compute_laplacian(hg, lap_type):
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
    
    hg_lape = encoder.add_laplacian_encodings(hg.copy(), type=lap_type, verbose=False)
    return hg_lape, L