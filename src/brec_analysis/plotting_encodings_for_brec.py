"""Plotting functions for the encodings of the BREC dataset"""

import os

import matplotlib.pyplot as plt
import numpy as np


def save_comparison_plot(
    plt,
    plot_dir: str,
    pair_idx: str | float | None,
    category: str,
    name_of_encoding: str,
    k: int,
):
    """Helper function to save the comparison plot.

    Args:
        plt:
            matplotlib plot
        plot_dir:
            directory to save the plot
        pair_idx:
            index of the pair
        category:
            category of the pair
        name_of_encoding:
            name of the encoding
    """
    os.makedirs(plot_dir, exist_ok=True)
    if "lape" in name_of_encoding.lower() or "rwpe" in name_of_encoding.lower():
        filename_base = (
            f"pair_{pair_idx}_k_{k}_{category.lower()}"
            if pair_idx is not None
            else "comparison"
        )
    else:
        filename_base = (
            f"pair_{pair_idx}_{category.lower()}"
            if pair_idx is not None
            else "comparison"
        )

    print(
        f"Saving plot to {plot_dir}/{filename_base}_{name_of_encoding.lower()}_comparison.png"
    )
    plt.savefig(
        f"{plot_dir}/{filename_base}_{name_of_encoding.lower()}_comparison.png",
        bbox_inches="tight",
        dpi=300,
    )


def plot_matched_encodings(
    is_direct_match: bool,
    is_same_up_to_scaling: bool,
    scaling_factor: float,
    permuted: np.ndarray,
    perm: tuple[int, ...],
    encoding1: np.ndarray,
    encoding2: np.ndarray,
    name1: str = "Graph A",
    name2: str = "Graph B",
    title: str = "",
    graph_type: str = "Graph",
    k: int = 1,
) -> None:
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

    # Create figure and axes
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 30))

    # do abs values here:
    encoding1 = np.abs(encoding1)
    encoding2 = np.abs(encoding2)

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
        if is_direct_match:
            diff = np.where(np.abs(diff) < 1e-10, 0, diff)
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
        title = f"{graph_type} {title} - {k} \n " + "\n".join(match_status)
    else:
        title = f"{graph_type} - {k} \n " + "\n".join(match_status)
    plt.suptitle(title, y=1.05)
