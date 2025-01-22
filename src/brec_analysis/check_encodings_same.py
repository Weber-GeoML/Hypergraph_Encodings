"""Functions for checking if two encodings are the same"""

import matplotlib.pyplot as plt
import numpy as np

from brec_analysis.add_encodings import get_encodings
from brec_analysis.laplacians_specific_functions import (
    check_isospectrality,
    compute_laplacian,
)
from brec_analysis.match_encodings import (
    check_encodings_same_up_to_scaling,
    find_encoding_match,
)
from brec_analysis.plotting_encodings_for_brec import save_comparison_plot
from brec_analysis.plotting_graphs_and_hgraphs_for_brec import plot_matched_encodings
from brec_analysis.printers_and_loggers import print_comparison_results
from brec_analysis.utils_for_brec import create_comparison_result
from encodings_hnns.encodings import HypergraphEncodings


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
        if not is_match:
            print("**-" * 20)
            print(f"We are also checking up to scaling for {name_of_encoding}")
            (
                is_same_up_to_scaling,
                scaling_factor,
                perm_up_to_scaling,
                permuted_up_to_scaling,
            ) = check_encodings_same_up_to_scaling(
                eigenvectors1, eigenvectors2, verbose=False
            )
            if is_same_up_to_scaling and not np.isclose(
                scaling_factor, 1.0, rtol=1e-10
            ):
                # Only print if there's actually a non-trivial scaling
                print("⛔️ The encodings are the same up to scaling")
                print(f"The scaling factor is {scaling_factor}")
            print("**-" * 20)
        plot_matched_encodings(
            is_match,
            is_same_up_to_scaling,
            scaling_factor,
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
                hg1, encoder_shrikhande, name_of_encoding, k_rwpe=k, k_lape=k
            )
            hg2_encodings = get_encodings(
                hg2, encoder_rooke, name_of_encoding, k_rwpe=k, k_lape=k
            )

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
            if not is_direct_match:
                print("**-" * 20)
                print(f"We are also checking up to scaling for {name_of_encoding}")
                (
                    is_same_up_to_scaling,
                    scaling_factor,
                    perm_up_to_scaling,
                    permuted_up_to_scaling,
                ) = check_encodings_same_up_to_scaling(
                    hg1_encodings["features"],
                    hg2_encodings["features"],
                    verbose=False,
                )
                if is_same_up_to_scaling and not np.isclose(
                    scaling_factor, 1.0, rtol=1e-10
                ):
                    # Only print if there's actually a non-trivial scaling
                    print("⛔️ The encodings are the same up to scaling")
                    print(f"The scaling factor is {scaling_factor}")
                print("**-" * 20)
            plot_matched_encodings(
                is_direct_match,
                is_same_up_to_scaling,
                scaling_factor,
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
        hg1_encodings = get_encodings(
            hg1, encoder_shrikhande, name_of_encoding, k_rwpe=k, k_lape=k
        )
        hg2_encodings = get_encodings(
            hg2, encoder_rooke, name_of_encoding, k_rwpe=k, k_lape=k
        )

        if verbose:
            # print the feature name and the encoding name
            print(f"features: \n {hg1_encodings['features']}")
            print(f"features: \n {hg2_encodings['features']}")
            print(f"Encoding name: {name_of_encoding}")

        # Plot and get match results
        is_direct_match, permuted, perm = find_encoding_match(
            hg1_encodings["features"], hg2_encodings["features"]
        )
        if not is_direct_match:
            print("**-" * 20)
            print(f"We are also checking up to scaling for {name_of_encoding}")
            (
                is_same_up_to_scaling,
                scaling_factor,
                perm_up_to_scaling,
                permuted_up_to_scaling,
            ) = check_encodings_same_up_to_scaling(
                hg1_encodings["features"],
                hg2_encodings["features"],
                verbose=False,
            )
            if is_same_up_to_scaling and not np.isclose(
                scaling_factor, 1.0, rtol=1e-10
            ):
                # Only print if there's actually a non-trivial scaling
                print("⛔️ The encodings are the same up to scaling")
                print(f"The scaling factor is {scaling_factor}")
            print("**-" * 20)
        plot_matched_encodings(
            is_direct_match,
            is_same_up_to_scaling,
            scaling_factor,
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
