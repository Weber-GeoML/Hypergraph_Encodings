"""Functions for printing and logging results"""

import numpy as np


def print_comparison_results(
    is_match: bool,
    name_of_encoding: str,
    perm: np.ndarray,
    permuted: np.ndarray,
    permuted2: np.ndarray,
    hg1_encodings: dict,
    hg2_encodings: dict,
) -> None:
    """Helper function to print comparison results.

    Args:
        is_match:
            whether the encodings are the same
        name_of_encoding:
            the name of the encoding
        perm:
            the permutation that was applied
        permuted:
            the permuted encoding
        permuted2:
            the permuted encoding of the second graph
        hg1_encodings:
            the encodings of the first graph
        hg2_encodings:
            the encodings of the second graph
    """

    if is_match:
        print(f"\n✅ Found matching permutation for {name_of_encoding}!")
        print(f"Permutation: {perm}")
        print("Statistics after permutation:")
        max_diff = np.max(np.abs(permuted - permuted2))
        print(f"Max difference: {max_diff}")
        # assert np.isclose(max_diff, 0, rtol=1e-9)
        mean_diff = np.mean(np.abs(permuted - permuted2))
        # print(
        #     f"Mean difference: {mean_diff}"
        # )

        # assert np.isclose(mean_diff, 0, rtol=1e-9)
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
