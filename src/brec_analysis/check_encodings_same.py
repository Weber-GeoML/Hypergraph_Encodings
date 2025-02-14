"""Functions for checking if two encodings are the same."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from brec_analysis.add_encodings import get_encodings
from brec_analysis.laplacians_specific_functions import (
    check_isospectrality,
    compute_laplacian,
)
from brec_analysis.match_encodings import find_encoding_match
from brec_analysis.match_status import MatchStatus
from brec_analysis.plotting_encodings_for_brec import (
    plot_matched_encodings,
    save_comparison_plot,
)
from brec_analysis.printers_and_loggers import print_comparison_results
from encodings_hnns.encodings import HypergraphEncodings

# TODO: clean up


def lap_checks_to_clean_up(name_of_encoding: str, hg1, hg2, graph_type: str):
    """Function to clean up.

    Args:
        name_of_encoding:
            name of the encoding
        hg1:
            first graph
        hg2:
            second graph
        graph_type:
            type of the graph (hypergraph or graph)

    Returns:
        tuple:
            eigenvectors for the first and second graph
    """
    # Handle Laplacian encodings
    lap_type = name_of_encoding.split("-")[1]  # Get Normalized, RW, or Hodge
    print(f"Computing Laplacian for {lap_type}")
    # Get Laplacian matrices and features
    _, L1 = compute_laplacian(hg1, lap_type)
    _, L2 = compute_laplacian(hg2, lap_type)

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
        print(f"The two graphs have different eigenvector norms for {name_of_encoding}")
        same_properties = False

    # Print comparison of norms
    verbose = False
    if verbose:
        print("\nComparison of eigenvector norms:")
        for name in ["Graph A", "Graph B"]:
            print(f"{name} Laplacian eigenvector norms: {properties[name]['norms']}")

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

    return eigenvectors1, eigenvectors2


def get_modified_name(name: str, k: int) -> str:
    """Handle k-dependent naming."""
    if name in ["RWPE", "LAPE-RW"]:
        return f"{name}-k{k}"
    return name


def get_appropriate_encodings(
    name: str,
    hg1,
    hg2,
    encoder1,
    encoder2,
    k: int,
    verbose: bool = False,
) -> tuple:
    """Get the appropriate encodings based on type.

    Args:
        name:
            name of the encoding
        hg1:
            first graph
        hg2:
            second graph
        encoder1:
            encoder for the first graph
        encoder2:
            encoder for the second graph
        k:
            k value for the encodings
        verbose:
            whether to print verbose output

    Returns:
        tuple:
            encodings for the first and second graph
    """
    return get_regular_encodings(name, hg1, hg2, encoder1, encoder2, k, verbose=verbose)


def get_laplacian_encodings(name: str, hg1, hg2, k: int) -> tuple:
    """Get Laplacian-based encodings.

    Args:
        name:
            name of the encoding
        hg1:
            first graph
        hg2:
            second graph

    Returns:
        tuple:
            encodings for the first and second graph
    """
    lap_type = name.split("-")[1]
    _, L1 = compute_laplacian(hg1, lap_type)
    _, L2 = compute_laplacian(hg2, lap_type)

    _, eigenvectors1 = np.linalg.eigh(L1)
    _, eigenvectors2 = np.linalg.eigh(L2)
    eigenvectors1 = eigenvectors1[:, :k]
    eigenvectors2 = eigenvectors2[:, :k]

    return eigenvectors1, eigenvectors2


def get_regular_encodings(
    name: str,
    hg1: nx.Graph,
    hg2: nx.Graph,
    encoder1: HypergraphEncodings,
    encoder2: HypergraphEncodings,
    k: int,
    verbose: bool = False,
) -> tuple:
    """Get regular (non-Laplacian) encodings.

    Args:
        name:
            name of the encoding
        hg1:
            first graph
        hg2:
            second graph
        encoder1:
            encoder for the first graph
        encoder2:
            encoder for the second graph
        k:
            k value for the encodings

    Returns:
        tuple:
            encodings for the first and second graph. ONLY THE FEATURES ARE RETURNED
    """
    hg1_copy = hg1.copy()
    hg2_copy = hg2.copy()
    hg1_encodings = get_encodings(hg1_copy, encoder1, name, k_rwpe=k, k_lape=k)
    hg2_encodings = get_encodings(hg2_copy, encoder2, name, k_rwpe=k, k_lape=k)
    assert hg1_encodings is not None
    assert hg2_encodings is not None
    if verbose:
        print(f"hg1_encodings for {name}{k}: \n {hg1_encodings['features']}")
        print(f"hg2_encodings for {name}{k}: \n {hg2_encodings['features']}")
    return hg1_encodings["features"], hg2_encodings["features"]


def checks_encodings(
    name_of_encoding: str,
    hg1,
    hg2,
    encoder_number_one: HypergraphEncodings,
    encoder_number_two: HypergraphEncodings,
    name1: str = "Graph A",
    name2: str = "Graph B",
    save_plots: bool = True,
    plot_dir: str = "plots/encodings",
    pair_idx: int | str | None = None,
    category: str | None = None,
    is_isomorphic: bool | None = None,
    node_mapping: dict | None = None,
    graph_type: str = "Graph",
    k: int = 3,
    verbose: bool = False,
) -> dict:
    """Check if two graphs have the same encodings. Returns comparison results.

    Args:
        name_of_encoding:
            name of the encoding
        hg1:
            first graph
        hg2:
            second graph
        encoder_number_one:
            encoder for the first graph
        encoder_number_two:
            encoder for the second graph
        name1:
            name of the first graph
        name2:
            name of the second graph
        save_plots:
            whether to save the plots
        plot_dir:
            directory to save the plots
        pair_idx:
            index of the pair
        category:
            category of the pair
        is_isomorphic:
            whether the graphs are isomorphic
        node_mapping:
            node mapping between the two graphs
        graph_type:
            type of the graph
        k:
            k value for the encodings
        verbose:
            whether to print verbose output

    Returns:
        the comparison results
    """

    assert hg1 is not None
    assert hg2 is not None
    assert encoder_number_one is not None
    assert encoder_number_two is not None

    # Initialize result dictionary
    comparison_result: dict = {
        "status": None,
        "scaling_factor": None,
        "permutation": None,
    }

    # The name_of_encoding might have been modified to include k
    modified_name = name_of_encoding
    modified_name = get_modified_name(name_of_encoding, k)
    print(f"Modified name: {modified_name}")

    hg1_copy = hg1.copy()
    hg2_copy = hg2.copy()

    # Get encodings based on type
    hg1_encodings, hg2_encodings = get_appropriate_encodings(
        name_of_encoding,
        hg1_copy,
        hg2_copy,
        encoder_number_one,
        encoder_number_two,
        k,
    )
    if graph_type == "hypergraph" and "lape" in name_of_encoding.lower():
        # remove the first column
        print(f"Removing first column of {name_of_encoding} for type {graph_type}")
        hg1_encodings = hg1_encodings[:, 1:]
        hg2_encodings = hg2_encodings[:, 1:]

    print(f"hg1_encodings: \n {hg1_encodings.shape}")
    print(f"hg2_encodings: \n {hg2_encodings.shape}")
    # assert False

    assert hg1_encodings is not None
    assert hg2_encodings is not None

    if verbose:
        # print the feature name and the encoding name
        print(f"features: \n {hg1_encodings}")
        print(f"features: \n {hg2_encodings}")
        print(f"Encoding name: {name_of_encoding}")

    print(f"Encoding name: {name_of_encoding}")
    match_result = check_for_matches(hg1_encodings, hg2_encodings, name_of_encoding)
    if match_result["status"] == MatchStatus.TIMEOUT:
        print("🚨 Timeout")
    comparison_result.update(match_result)

    if match_result["status"] != MatchStatus.TIMEOUT:
        if match_result["is_direct_match"]:
            # Plot and get match results
            plot_matched_encodings(
                match_result["is_direct_match"],
                match_result["is_same_up_to_scaling"],
                match_result["scaling_factor"],
                match_result["permuted"],
                match_result["permutation"],
                match_result["permuted"],
                hg2_encodings,
                name1,
                name2,
                modified_name,  # Pass modified name as title
                graph_type,
                k,
            )
        else:
            # Plot and get match results
            plot_matched_encodings(
                match_result["is_direct_match"],
                match_result["is_same_up_to_scaling"],
                match_result["scaling_factor"],
                match_result["permuted"],
                match_result["permutation"],
                hg1_encodings,
                hg2_encodings,
                name1,
                name2,
                modified_name,  # Pass modified name as title
                graph_type,
                k,
            )

        # if match_result["is_direct_match"]:
        #     print(match_result["status"])
        #     assert np.allclose(
        #         match_result["permuted"], match_result["permuted2"], rtol=1e-9
        #     )  # type: ignore

        # Print results
        print_comparison_results(
            match_result["is_direct_match"],
            name_of_encoding,
            match_result["permutation"],
            match_result["permuted"],
            match_result["permuted2"],
            {"features": hg1_encodings},
            {"features": hg2_encodings},
        )

    if name_of_encoding.startswith("LAPE-"):
        # eigenvectors1, eigenvectors2 = lap_checks_to_clean_up(
        #     name_of_encoding, hg1, hg2, graph_type
        # )
        # assert np.isclose(eigenvectors1[:, :k], hg1_encodings).all()
        # assert np.isclose(eigenvectors2[:, :k], hg2_encodings).all()

        # debug = False
        # if debug:
        # # TODO: cleaan up
        # print("*" * 100)
        # print(f"DEBUG: {graph_type}")
        # print("*" * 100)

        # # Handle encodings
        # hg1_encodings = get_encodings(
        #     hg1, encoder_number_one, name_of_encoding, k_rwpe=k, k_lape=k
        # )
        # hg2_encodings = get_encodings(
        #     hg2, encoder_number_two, name_of_encoding, k_rwpe=k, k_lape=k
        # )

        # keep_first_column = True

        # if keep_first_column:
        #     print(f"Truncating encodings to first 2 columns at {graph_type}")

        #     # Keep the first 2 columns od each encoding
        #     hg1_encodings = hg1_encodings[:, :1]
        #     hg2_encodings = hg2_encodings[:, :1]

        # print(f"features: \n {hg1_encodings}")
        # print(f"features: \n {hg2_encodings}")

        # # Plot and get match results
        # is_direct_match, permuted, perm, timeout= find_encoding_match(
        #     hg1_encodings, hg2_encodings
        # )
        # if not is_direct_match:
        #     print("**-" * 20)
        #     print(f"We are also checking up to scaling for {name_of_encoding}")
        #     (
        #         is_same_up_to_scaling,
        #         scaling_factor,
        #         perm_up_to_scaling,
        #         permuted_up_to_scaling,
        #         permuted2_up_to_scaling,
        #     ) = check_encodings_same_up_to_scaling(
        #         hg1_encodings,
        #         hg2_encodings,
        #         name_of_encoding=name_of_encoding,
        #         verbose=False,
        #     )
        #     if is_same_up_to_scaling and not np.isclose(
        #         scaling_factor, 1.0, rtol=1e-10
        #     ):
        #         # Only print if there's actually a non-trivial scaling
        #         print("⛔️ The encodings are the same up to scaling")
        #         print(f"The scaling factor is {scaling_factor}")
        #     print("**-" * 20)
        # else:
        #     is_same_up_to_scaling = True
        #     scaling_factor = 1.0

        # plot_matched_encodings(
        #     is_direct_match,
        #     is_same_up_to_scaling,
        #     scaling_factor,
        #     permuted,
        #     perm,
        #     hg1_encodings,
        #     hg2_encodings,
        #     name1,
        #     name2,
        #     modified_name,  # Pass modified name as title
        #     graph_type,
        # )

        # # Print results
        # print_comparison_results(
        #     is_direct_match,
        #     name_of_encoding,
        #     perm,
        #     permuted,
        #     {"features": hg1_encodings},
        #     {"features": hg2_encodings},
        # )

        # print("*" * 100)
        # print(f"END DEBUG: {graph_type}")
        # print("*" * 100)

        # Save plot if requested - This will handle both LAPE and non-LAPE cases
        if save_plots:
            if pair_idx is None:
                pair_idx_str = "Example_pair"
            else:
                pair_idx_str = str(pair_idx)
            assert category is not None
            save_comparison_plot(
                plt, plot_dir, pair_idx_str, category, modified_name, k
            )
            plt.close()  # Only close the figure once at the end

    return comparison_result


def check_for_matches(encoding1, encoding2, name: str) -> dict:
    """Check for direct matches and scaling matches.

    Args:
        encoding1:
            the first encoding
        encoding2:
            the second encoding
        name:
            the name of the encoding

    Returns:
        the comparison results
    """
    is_direct_match: bool | None = None
    permuted: np.ndarray | None = None
    permuted2: np.ndarray | None = None
    perm: tuple[int, ...] | None = None
    result = find_encoding_match(
        encoding1, encoding2, name_of_encoding=name, verbose=True
    )
    assert result is not None
    is_direct_match, permuted, perm, permuted2, _ = result

    is_same_up_to_scaling: bool = False
    scaling_factor: float | None = None

    assert encoding1 is not None
    assert encoding2 is not None

    # skipping for speed.
    # if not is_direct_match:
    #     print("**-" * 20)
    #     print(f"We are also checking up to scaling for {name}")
    #     is_same_up_to_scaling, scaling_factor, perm, permuted, permuted2 = (
    #         check_encodings_same_up_to_scaling(encoding1, encoding2, name_of_encoding=name_of_encoding, verbose=False)
    #     )
    #     if is_same_up_to_scaling and not np.isclose(scaling_factor, 1.0, rtol=1e-10):
    #         # Only print if there's actually a non-trivial scaling
    #         print("⛔️ The encodings are the same up to scaling")
    #         print(f"The scaling factor is {scaling_factor}")
    #     print("**-" * 20)
    # else:
    #     is_same_up_to_scaling = True
    #     scaling_factor = 1.0

    status = MatchStatus.NO_MATCH
    if is_direct_match:
        status = MatchStatus.EXACT_MATCH
        # if timeout == "timeout":
        #     status = MatchStatus.TIMEOUT
    elif is_same_up_to_scaling:
        status = MatchStatus.SCALED_MATCH

    return {
        "status": status,
        "is_direct_match": is_direct_match,
        "is_same_up_to_scaling": is_same_up_to_scaling,
        "scaling_factor": scaling_factor if is_same_up_to_scaling else None,
        "permutation": perm,
        "permuted": permuted,
        "permuted2": permuted2,
    }
