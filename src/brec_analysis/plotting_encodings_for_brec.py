"""Plotting functions for the encodings of the BREC dataset"""

import os

import matplotlib.pyplot as plt



def save_comparison_plot(
    plt: plt.Axes,
    plot_dir: str,
    pair_idx: str,
    category: str,
    name_of_encoding: str,
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
    filename_base = (
        f"pair_{pair_idx}_{category.lower()}" if pair_idx is not None else "comparison"
    )
    plt.savefig(
        f"{plot_dir}/{filename_base}_{name_of_encoding.lower()}_comparison.png",
        bbox_inches="tight",
        dpi=300,
    )
