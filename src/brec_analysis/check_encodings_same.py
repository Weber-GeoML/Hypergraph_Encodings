"""Functions for checking if two encodings are the same"""

from itertools import permutations

import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
from torch_geometric.data import Data

from brec_analysis.laplacians_specific_functions import (
    check_isospectrality,
    compute_laplacian,
    reconstruct_matrix,
)
from brec_analysis.match_encodings import (
    check_encodings_same_up_to_scaling,
    find_encoding_match,
)
from brec_analysis.plotting_encodings_for_brec import save_comparison_plot
from brec_analysis.utils_for_brec import create_comparison_result
from encodings_hnns.encodings import HypergraphEncodings
from encodings_hnns.laplacians import Laplacians


def plot_matched_encodings(
    is_direct_match: bool,
    permuted: np.ndarray,
    perm: tuple[int, ...],
    encoding1: np.ndarray,
    encoding2: np.ndarray,
    ax1: plt.Axes,
    ax2: plt.Axes,
    ax3: plt.Axes,
    name1: str = "Graph A",
    name2: str = "Graph B",
    title: str = "",
    graph_type: str = "Graph",
) -> tuple[bool, np.ndarray, tuple[int, ...]]:
    """
    Plot two encodings and their difference, attempting to match their row orderings if possible.

    Args:
        encoding1, encoding2:
            numpy arrays of shape (n, d)
        ax1, ax2, ax3:
            matplotlib axes for plotting
        name1, name2:
            names of the graphs
        title:
            title for the plots
        graph_type:
            string indicating "Graph" or "Hypergraph"

    Returns:
        is_direct_match:
            whether the encodings are the same
        permuted:
            the permuted encoding of encoding1
        perm:
            the permutation that was applied
    """

    print("**-" * 20)
    if not is_direct_match:
        print(f"We are also checking up to scaling for {title}")
        (
            is_same_up_to_scaling,
            scaling_factor,
            perm_up_to_scaling,
            permuted_up_to_scaling,
        ) = check_encodings_same_up_to_scaling(encoding1, encoding2, verbose=False)
        if is_same_up_to_scaling and not np.isclose(scaling_factor, 1.0, rtol=1e-10):
            # Only print if there's actually a non-trivial scaling
            print("⛔️ The encodings are the same up to scaling")
            print(f"The scaling factor is {scaling_factor}")
        print("**-" * 20)

    if is_direct_match:
        vmin = min(np.min(permuted), np.min(encoding2))
        vmax = max(np.max(permuted), np.max(encoding2))
        im1 = ax1.imshow(permuted, cmap="viridis", vmin=vmin, vmax=vmax)
        im2 = ax2.imshow(encoding2, cmap="viridis", vmin=vmin, vmax=vmax)
        diff = np.abs(permuted - encoding2)
        # add the min and max value of the encoding to the title
        ax1.set_title(
            f"{name1}\n(Permuted to match {name2}) \n min: {np.min(permuted):.2e}, max: {np.max(permuted):.2e}"
        )
    else:
        vmin = min(np.min(encoding1), np.min(encoding2))
        vmax = max(np.max(encoding1), np.max(encoding2))
        im1 = ax1.imshow(encoding1, cmap="viridis", vmin=vmin, vmax=vmax)
        im2 = ax2.imshow(encoding2, cmap="viridis", vmin=vmin, vmax=vmax)
        diff = encoding1 - encoding2
        # add the min and max value of the encoding to the title
        ax1.set_title(
            f"{name1}\n(Original ordering) \n min: {np.min(encoding1):.4e}, max: {np.max(encoding1):.4e}"
        )

    # Plot difference matrix
    im3 = ax3.imshow(
        diff, cmap="Blues"
    )  # Using Blues colormap to highlight differences

    # add the min and max value of the encoding two to the title
    ax2.set_title(
        f"{name2}\n(min: {np.min(encoding2):.2e}, max: {np.max(encoding2):.2e})"
    )

    # Check if difference is uniformly zero
    if np.allclose(diff, np.zeros_like(diff)):
        # replace any value whose absolute value is less than 1e-13 with 0
        diff = np.where(np.abs(diff) < 1e-13, 0, diff)
        ax3.set_title("Absolute Difference\n(Uniformly Zero)")
    else:
        # Get max absolute values for both encodings
        max_abs1 = np.max(np.abs(encoding1))
        max_abs2 = np.max(np.abs(encoding2))
        ax3.set_title(
            f"Difference\nMax abs values: {max_abs1:.4e} vs {max_abs2:.4e}\n Mean abs values: {np.mean(np.abs(encoding1)):.4e} vs {np.mean(np.abs(encoding2)):.4e} \n Min abs values: {np.min(np.abs(encoding1)):.4e} vs {np.min(np.abs(encoding2)):.4e}"
        )

    # Add colorbars
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    plt.colorbar(im3, ax=ax3)

    # Add row labels if the matrices are small enough
    if encoding1.shape[0] <= 10:
        # For the first plot, use permuted node ordering if a match was found
        if is_direct_match:
            for i, p in enumerate(perm):
                ax1.text(-0.5, i, f"Node {p}", va="center")
        else:
            for i in range(encoding1.shape[0]):
                ax1.text(-0.5, i, f"Node {i}", va="center")

        # For second plot, always use original ordering
        for i in range(encoding2.shape[0]):
            ax2.text(-0.5, i, f"Node {i}", va="center")
            ax3.text(-0.5, i, f"Node {i}", va="center")

    # Determine match status
    match_status = []
    if is_direct_match:
        match_status.append(r"${ \bf [MATCH]}$")
    elif is_same_up_to_scaling:
        match_status.append(r"${ \bf [SCALED\ MATCH]}$")
        scale_info = f" (scaled by {scaling_factor:.2e})"
        match_status.append(scale_info)
    else:
        match_status.append(r"${\bf [NO\ MATCH]}$")

    # Add match status to the main title
    if title:
        title = f"{graph_type} {title} \n " + "\n".join(match_status)
    else:
        title = f"{graph_type} \n " + "\n".join(match_status)
    plt.suptitle(title, y=1.05)

    return is_direct_match, permuted, perm


def checks_encodings(
    name_of_encoding: str,
    hg1,
    hg2,
    encoder_shrikhande: HypergraphEncodings,
    encoder_rooke: HypergraphEncodings,
    name1: str = "Graph A",
    name2: str = "Graph B",
    save_plots: bool = True,
    plot_dir: str = "plots/encodings",
    pair_idx: int = None,
    category: str = None,
    is_isomorphic: bool = None,
    node_mapping: dict = None,
    graph_type: str = "Graph",
    k: int = 3,
    verbose: bool = False,
) -> dict:
    """Check if two graphs have the same encodings. Returns comparison results."""

    comparison_result = {}

    # Create figure and axes
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # The name_of_encoding might have been modified to include k
    modified_name = name_of_encoding
    if name_of_encoding in ["RWPE", "LAPE-RW"]:
        modified_name = f"{name_of_encoding}-k{k}"

    if name_of_encoding.startswith("LAPE-"):
        # Handle Laplacian encodings
        lap_type = name_of_encoding.split("-")[1]  # Get Normalized, RW, or Hodge
        print(f"Computing Laplacian for {lap_type}")
        # Get Laplacian matrices and features
        hg1_lape, L1 = compute_laplacian(hg1, lap_type)
        hg2_lape, L2 = compute_laplacian(hg2, lap_type)

        # Compute eigendecomposition of Laplacian matrices
        eigenvalues1, eigenvectors1 = np.linalg.eigh(
            L1
        )  # Using eigh for symmetric matrices
        eigenvalues2, eigenvectors2 = np.linalg.eigh(L2)

        # Verify eigenvalues are sorted
        assert np.allclose(
            eigenvalues1, np.sort(eigenvalues1)
        ), "Eigenvalues of Graph A are not in order"
        assert np.allclose(
            eigenvalues2, np.sort(eigenvalues2)
        ), "Eigenvalues of Graph B are not in order"

        # Check matrix properties
        properties = {
            "Graph A": {
                "min_eigenvalue": np.min(eigenvalues1),
                "max_eigenvalue": np.max(eigenvalues1),
                "min_eigenvector": np.min(eigenvectors1),
                "rank": np.linalg.matrix_rank(eigenvectors1),
                "norms": np.sort(np.linalg.norm(eigenvectors1, axis=1)),
            },
            "Graph B": {
                "min_eigenvalue": np.min(eigenvalues2),
                "max_eigenvalue": np.max(eigenvalues2),
                "min_eigenvector": np.min(eigenvectors2),
                "rank": np.linalg.matrix_rank(eigenvectors2),
                "norms": np.sort(np.linalg.norm(eigenvectors2, axis=1)),
            },
        }

        # Initialize result flags
        same_properties = True

        # Compare properties
        for prop in ["rank", "min_eigenvalue", "max_eigenvalue"]:
            if not np.allclose(
                properties["Graph A"][prop],
                properties["Graph B"][prop],
                rtol=1e-10,
            ):
                print(f"The two graphs have different {prop} for {name_of_encoding}")
                same_properties = False

        # Compare norms
        same_norms = np.allclose(
            properties["Graph A"]["norms"],
            properties["Graph B"]["norms"],
            atol=1e-12,
            rtol=1e-12,
        )
        if not same_norms:
            print(
                f"The two graphs have different eigenvector norms for {name_of_encoding}"
            )
            same_properties = False

        # Print comparison of norms
        verbose = False
        if verbose:
            print("\nComparison of eigenvector norms:")
            for name in ["Graph A", "Graph B"]:
                print(
                    f"{name} Laplacian eigenvector norms: {properties[name]['norms']}"
                )

        # Try to find matching permutation for eigenvectors
        is_match, permuted, perm = find_encoding_match(eigenvectors1, eigenvectors2)
        plot_matched_encodings(
            is_match,
            permuted,
            perm,
            eigenvectors1,
            eigenvectors2,
            ax1,
            ax2,
            ax3,
            name1,
            name2,
            modified_name,
            "Hypergraph",
        )

        # Check for scaled match
        is_same_up_to_scaling, scaling_factor, _, _ = (
            check_encodings_same_up_to_scaling(
                eigenvectors1, eigenvectors2, verbose=False
            )
        )

        # Print results
        print_comparison_results(
            is_match,
            name_of_encoding,
            perm,
            permuted,
            {"features": eigenvectors1},
            {"features": eigenvectors2},
        )

        # if save_plots:
        #     # Additional plots
        #     # Features difference plot
        #     plt.figure(figsize=(8, 6))
        #     plt.imshow(hg1_lape["features"] - hg2_lape["features"], cmap="Blues")
        #     plt.colorbar()
        #     plt.title(f"Difference in {lap_type} Features\n{category} - Pair {pair_idx}")
        #     save_comparison_plot(plt, plot_dir, pair_idx, category, f"{name_of_encoding}_features")
        #     plt.close()

        #     # Laplacian matrices difference plot
        #     plt.figure(figsize=(8, 6))
        #     plt.imshow(L1 - L2, cmap="Blues")
        #     plt.colorbar()
        #     plt.title(f"Difference in {lap_type} Laplacian Matrices\n{category} - Pair {pair_idx}")
        #     save_comparison_plot(plt, plot_dir, pair_idx, category, f"{name_of_encoding}_matrices")
        #     plt.close()

        # Check isospectrality
        are_isospectral = check_isospectrality(eigenvalues1, eigenvalues2)
        if not are_isospectral:
            print(
                f"\n🚫 The two graphs are not isospectral for {name_of_encoding} at {graph_type}"
            )
        else:
            print(f"\n🟢 The two graphs are isospectral for {name_of_encoding}")
        # print is same properties with box
        print(
            f"\n{'🟢' if same_properties else '⛔️'} Properties comparison for {name_of_encoding} at {graph_type} \n"
        )

        # Store results for eigenvalues and eigenvectors
        comparison_result["eigenvalues"] = {"is_isospectral": are_isospectral}
        comparison_result["eigenvectors"] = create_comparison_result(
            is_match,
            is_same_up_to_scaling,
            scaling_factor if is_same_up_to_scaling else None,
        )
        comparison_result["properties"] = {"same": same_properties}

        #
        debug = False
        if debug:
            # TODO: cleaan up
            print("*" * 100)
            print(f"DEBUG: {graph_type}")
            print("*" * 100)

            # Handle encodings
            hg1_encodings = get_encodings(
                hg1, encoder_shrikhande, name_of_encoding, k=k
            )
            hg2_encodings = get_encodings(hg2, encoder_rooke, name_of_encoding, k=k)

            keep_first_column = True

            if keep_first_column:
                print(f"Truncating encodings to first 2 columns at {graph_type}")

                # Keep the first 2 columns od each encoding
                hg1_encodings["features"] = hg1_encodings["features"][:, :1]
                hg2_encodings["features"] = hg2_encodings["features"][:, :1]

            print(f"features: \n {hg1_encodings['features']}")
            print(f"features: \n {hg2_encodings['features']}")

            # Plot and get match results
            is_direct_match, permuted, perm = find_encoding_match(
                hg1_encodings["features"], hg2_encodings["features"]
            )
            plot_matched_encodings(
                is_direct_match,
                permuted,
                perm,
                hg1_encodings["features"],
                hg2_encodings["features"],
                ax1,
                ax2,
                ax3,
                name1,
                name2,
                modified_name,  # Pass modified name as title
                graph_type,
            )

            # Check for scaled match
            is_scaled_match, scaling_factor, _, _ = check_encodings_same_up_to_scaling(
                hg1_encodings["features"],
                hg2_encodings["features"],
                verbose=False,
            )

            # Print results
            print_comparison_results(
                is_direct_match,
                name_of_encoding,
                perm,
                permuted,
                {"features": hg1_encodings["features"]},
                {"features": hg2_encodings["features"]},
            )

            print("*" * 100)
            print(f"END DEBUG: {graph_type}")
            print("*" * 100)

    else:
        # Handle other encodings
        hg1_encodings = get_encodings(hg1, encoder_shrikhande, name_of_encoding, k=k)
        hg2_encodings = get_encodings(hg2, encoder_rooke, name_of_encoding, k=k)

        if verbose:
            # print the feature name and the encoding name
            print(f"features: \n {hg1_encodings['features']}")
            print(f"features: \n {hg2_encodings['features']}")
            print(f"Encoding name: {name_of_encoding}")

        # Plot and get match results
        is_direct_match, permuted, perm = find_encoding_match(
            hg1_encodings["features"], hg2_encodings["features"]
        )
        plot_matched_encodings(
            is_direct_match,
            permuted,
            perm,
            hg1_encodings["features"],
            hg2_encodings["features"],
            ax1,
            ax2,
            ax3,
            name1,
            name2,
            modified_name,  # Pass modified name as title
            graph_type,
        )

        # Check for scaled match
        is_scaled_match, scaling_factor, _, _ = check_encodings_same_up_to_scaling(
            hg1_encodings["features"],
            hg2_encodings["features"],
            verbose=False,
        )

        # Print results
        print_comparison_results(
            is_direct_match,
            name_of_encoding,
            perm,
            permuted,
            {"features": hg1_encodings["features"]},
            {"features": hg2_encodings["features"]},
        )

        comparison_result["features"] = create_comparison_result(
            is_direct_match,
            is_scaled_match,
            scaling_factor if is_scaled_match else None,
        )

    # Save plot if requested - This will handle both LAPE and non-LAPE cases
    if save_plots:
        plt.tight_layout()
        save_comparison_plot(plt, plot_dir, pair_idx, category, modified_name)
    plt.close()  # Only close the figure once at the end

    return comparison_result


def get_encodings(
    hg: Data, encoder: HypergraphEncodings, name_of_encoding: str, k: int = 1
) -> dict:
    """Helper function to get the appropriate encodings based on type.

    Args:
        hg:
            the hypergraph
        encoder:
            the encoder
        name_of_encoding:
            the name of the encoding
        k:
            the k value for the encoding

    Returns:
        the encodings
    """
    if name_of_encoding == "LDP":
        return encoder.add_degree_encodings(hg.copy(), verbose=False)
    elif name_of_encoding == "RWPE":
        # Add k to the name for random walks
        # TODO: fix two things. First, there are ones that should not be there.
        # TODO Two: pass in k through the all pipeline
        name_of_encoding = f"RWPE-k{k}"
        print(f"Adding random walk encodings with k={k} for {name_of_encoding}")
        print(f"features: \n {hg['features']}")
        return encoder.add_randowm_walks_encodings(
            hg.copy(), rw_type="WE", verbose=False, k=k
        )
    elif name_of_encoding == "LCP-ORC":
        return encoder.add_curvature_encodings(hg.copy(), verbose=False, type="ORC")
    elif name_of_encoding == "LCP-FRC":
        return encoder.add_curvature_encodings(hg.copy(), verbose=False, type="FRC")
    elif name_of_encoding == "LAPE-Normalized":
        return encoder.add_laplacian_encodings(
            hg.copy(), type="Normalized", verbose=False, use_same_sign=True
        )
    elif name_of_encoding == "LAPE-RW":
        # Add k to the name for random walk Laplacian
        name_of_encoding = f"LAPE-RW-k{k}"
        return encoder.add_laplacian_encodings(hg.copy(), type="RW", verbose=False, k=k)
    elif name_of_encoding == "LAPE-Hodge":
        return encoder.add_laplacian_encodings(
            hg.copy(), type="Hodge", verbose=False, use_same_sign=True
        )

    return None


def print_comparison_results(
    is_match: bool,
    name_of_encoding: str,
    perm: np.ndarray,
    permuted: np.ndarray,
    hg1_encodings: dict,
    hg2_encodings: dict,
):
    """Helper function to print comparison results."""
    # Check if we're dealing with Laplacian encodings
    is_laplacian = name_of_encoding.startswith("LAPE-")

    if is_match:
        print(f"\n✅ Found matching permutation for {name_of_encoding}!")
        print(f"Permutation: {perm}")
        print("Statistics after permutation:")
        print(f"Max difference: {np.max(np.abs(permuted - hg2_encodings['features']))}")
        print(
            f"Mean difference: {np.mean(np.abs(permuted - hg2_encodings['features']))}"
        )
        if not is_laplacian:  # Only print extra newline for non-Laplacian encodings
            print("\n")
    else:
        print(f"\n❌ No matching permutation found for {name_of_encoding}")
        print("Differences in original ordering:")
        print(
            f"Max abs values: {np.max(np.abs(hg1_encodings['features']))} vs {np.max(np.abs(hg2_encodings['features']))}"
        )
        print(
            f"Min abs values: {np.min(np.abs(hg1_encodings['features']))} vs {np.min(np.abs(hg2_encodings['features']))}"
        )
        print(
            f"Mean abs values: {np.mean(np.abs(hg1_encodings['features']))} vs {np.mean(np.abs(hg2_encodings['features']))}"
        )
        if not is_laplacian:  # Only print extra newline for non-Laplacian encodings
            print("\n")


# def analyze_graph_pair(graph1, graph2, pair_idx : str, category : str, is_isomorphic : bool):
#     """Analyze a pair of graphs and store comparison results"""
#     results = {
#         "pair_idx": pair_idx,
#         "category": category,
#         "is_isomorphic": is_isomorphic,
#         "graph_level": {},
#         "hypergraph_level": {},
#     }

#     encoder1 = HypergraphEncodings()
#     encoder2 = HypergraphEncodings()

#     # List of encodings to check
#     k_dependent_encodings = ["RWPE", "LAPE-RW"]
#     k_values = [1, 2, 20]

#     base_encodings = ["LDP", "LCP-FRC", "LCP-ORC", "LAPE-Normalized", "LAPE-Hodge"]

#     # Generate all encodings with k values
#     encodings_to_check = base_encodings.copy()
#     for enc in k_dependent_encodings:
#         for k in k_values:
#             encodings_to_check.append(f"{enc}-k{k}")

#     # Check graph-level encodings
#     for encoding in encodings_to_check:
#         # Extract k value if present in encoding name
#         k = 1  # default value
#         if "-k" in encoding:
#             base_encoding, k = encoding.split("-k")
#             k = int(k)
#         else:
#             base_encoding = encoding

#         results["graph_level"][encoding] = checks_encodings(
#             base_encoding,
#             graph1,
#             graph2,
#             encoder1,
#             encoder2,
#             graph_type="Graph",
#             k=k,
#         )

#     # Check hypergraph-level encodings
#     for encoding in encodings_to_check:
#         # Extract k value if present in encoding name
#         k = 1  # default value
#         if "-k" in encoding:
#             base_encoding, k = encoding.split("-k")
#             k = int(k)
#         else:
#             base_encoding = encoding

#         results["hypergraph_level"][encoding] = checks_encodings(
#             base_encoding,
#             data1_lifted,
#             data2_lifted,
#             encoder1,
#             encoder2,
#             graph_type="Hypergraph",
#             k=k,
#         )

#     # Save results to JSON
#     os.makedirs("results/comparisons", exist_ok=True)
#     output_file = f"results/comparisons/pair_{pair_idx}_{category.lower()}.json"
#     with open(output_file, "w") as f:
#         json.dump(results, f, indent=2)

#     return results


def print_comparison_summary(results):
    """Print a human-readable summary of the comparison results"""
    print(f"\nSummary for Pair {results['pair_idx']} ({results['category']}):")
    print("-" * 50)

    for level in ["graph_level", "hypergraph_level"]:
        print(f"\n{level.replace('_', ' ').title()}:")
        print("-" * 30)

        # Group k-dependent encodings together
        grouped_results = {}
        for encoding, result in results[level].items():
            base_encoding = encoding.split("-k")[0] if "-k" in encoding else encoding
            if base_encoding not in grouped_results:
                grouped_results[base_encoding] = []
            grouped_results[base_encoding].append((encoding, result))

        # Print results with k-dependent encodings grouped
        for base_encoding, encoding_results in grouped_results.items():
            if len(encoding_results) == 1:
                # Single encoding (non k-dependent)
                encoding, result = encoding_results[0]
                status = result.get("features", {}).get("status", "N/A")
                scaling = result.get("features", {}).get("scaling_factor", None)

                status_str = status
                if status == "SCALED_MATCH" and scaling is not None:
                    status_str += f" (scale: {scaling:.2e})"

                print(f"{encoding:15} : {status_str}")

                # Print additional info for Laplacian encodings
                if encoding.startswith("LAPE-"):
                    print(
                        f"{'':15}   Isospectral: {result.get('eigenvalues', {}).get('is_isospectral', 'N/A')}"
                    )
            else:
                # k-dependent encodings
                print(f"\n{base_encoding} results:")
                for encoding, result in sorted(
                    encoding_results, key=lambda x: int(x[0].split("k")[-1])
                ):
                    status = result.get("features", {}).get("status", "N/A")
                    scaling = result.get("features", {}).get("scaling_factor", None)

                    status_str = status
                    if status == "SCALED_MATCH" and scaling is not None:
                        status_str += f" (scale: {scaling:.2e})"

                    k_value = encoding.split("k")[-1]
                    print(f"{'':2}k={k_value:3} : {status_str}")
