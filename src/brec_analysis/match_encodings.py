"""Functions for matching encodings.


Try to find a permutation of the rows of encoding1 that makes it equal to encoding2.
Also try up to scaling.
"""

from itertools import permutations

import numpy as np


def find_encoding_match(
    encoding1: np.ndarray, encoding2: np.ndarray, verbose: bool = True
) -> tuple[bool, np.ndarray | None, tuple[int, ...] | None, np.ndarray | None]:
    """
    Check if two encodings are equivalent under row permutations.
    Returns (is_match, permuted_encoding1, permutation) if found, (False, None, None) if not.

    Args:
        encoding1: numpy array of shape (n, d)
        encoding2: numpy array of shape (n, d)

    Returns:
        is_match:
            whether the two encodings are the same (up to row permutations)
        permuted_encoding1:
            the permuted encoding of encoding1
        permutation:
            the permutation that was applied
        permuted_encoding2:
            the permuted encoding of encoding2
    """
    if encoding1.shape != encoding2.shape:
        return False, None, None, None

    # First check if the encodings are identical
    if np.allclose(encoding1, encoding2, rtol=1e-13):
        # Return identity permutation if encodings are identical
        if verbose:
            print("Free lunch!")
        n_rows = encoding1.shape[0]
        return True, encoding1, tuple(range(n_rows)), encoding2

    # Pre-compute expensive operations
    abs_enc1 = np.abs(encoding1)
    abs_enc2 = np.abs(encoding2)

    # Check the max absolute value of each encodings. If they are different, return False
    if not np.isclose(np.max(abs_enc1), np.max(abs_enc2), rtol=1e-10):
        if verbose:
            print("Different because:")
            print(f"Max absolute value of encoding1: {np.max(np.abs(encoding1))}")
            print(f"Max absolute value of encoding2: {np.max(np.abs(encoding2))}")
            print("\n")
        return False, None, None, None

    # Same for min
    if not np.isclose(np.min(abs_enc1), np.min(abs_enc2), rtol=1e-10):
        if verbose:
            print("Different because:")
            print(f"Min absolute value of encoding1: {np.min(np.abs(encoding1))}")
            print(f"Min absolute value of encoding2: {np.min(np.abs(encoding2))}")
            print("\n")
        return False, None, None, None

    # Vectorized column comparisons for both max and min
    max_cols1 = np.max(abs_enc1, axis=0)
    max_cols2 = np.max(abs_enc2, axis=0)
    min_cols1 = np.min(abs_enc1, axis=0)
    min_cols2 = np.min(abs_enc2, axis=0)

    # Check max values
    if not np.allclose(max_cols1, max_cols2, rtol=1e-10):
        if verbose:
            diff_cols = ~np.isclose(max_cols1, max_cols2, rtol=1e-10)
            print(f"Different max at columns: {np.where(diff_cols)[0]}")
            print(f"Max values enc1: {max_cols1[diff_cols]}")
            print(f"Max values enc2: {max_cols2[diff_cols]}")
        return False, None, None, None

    # Check min values
    if not np.allclose(min_cols1, min_cols2, rtol=1e-10):
        if verbose:
            diff_cols = ~np.isclose(min_cols1, min_cols2, rtol=1e-10)
            print(f"Different min at columns: {np.where(diff_cols)[0]}")
            print(f"Min values enc1: {min_cols1[diff_cols]}")
            print(f"Min values enc2: {min_cols2[diff_cols]}")
        return False, None, None, None

    n_rows = encoding1.shape[0]

    # For small matrices, we can try all permutations
    if n_rows <= 10:  # Adjust this threshold based on your needs
        for perm_ in permutations(range(n_rows)):
            permuted = encoding1[list(perm_), :]
            if np.allclose(permuted, encoding2, rtol=1e-13):
                return True, permuted, tuple(perm_), encoding2
    else:
        # For larger matrices, use a heuristic approach based on row sorting
        # This works because:
        # 1. If two graphs are isomorphic, their encodings differ only by row permutation
        # 2. Lexicographical sorting will arrange rows in a canonical order
        # 3. After sorting, isomorphic graphs will have identical encodings

        # Sort rows lexicographically for both matrices
        # This creates a canonical form independent of original node ordering
        lexsort1: np.ndarray = np.lexsort(encoding1.T)
        lexsort2: np.ndarray = np.lexsort(encoding2.T)
        sorted1 = encoding1[lexsort1]
        sorted2 = encoding2[lexsort2]

        # Compare sorted matrices
        # If they're equal (up to numerical precision), the graphs are isomorphic
        # This is valid because:
        # - Isomorphic graphs must have the same multiset of row vectors
        # - Lexicographical sorting creates the same ordering for identical multisets
        if np.allclose(sorted1, sorted2, rtol=1e-13):
            # Find the permutation that was applied to the first encoding
            # This gives us the mapping between the original and sorted node orderings
            perm: np.ndarray = np.argsort(lexsort1)
            return True, sorted1, tuple(perm), sorted2

    return False, None, None, None


def check_encodings_same_up_to_scaling(
    encoding1: np.ndarray, encoding2: np.ndarray, verbose: bool = False
) -> tuple[
    bool, float | None, tuple[int, ...] | None, np.ndarray | None, np.ndarray | None
]:
    """
    Check if two encodings are equivalent under row permutations and scaling.

    Args:
        encoding1: numpy array of shape (n, d)
        encoding2: numpy array of shape (n, d)
        verbose: whether to print diagnostic information

    Returns:
        is_same:
            bool indicating if encodings are the same up to scaling and permutation
        scaling_factor:
            float, the scaling factor that makes them match (None if no match)
        permutation:
            the permutation that was applied (None if no match)
        permuted_encoding:
            the permuted and scaled encoding1 (None if no match)
        permuted_encoding2:
            the permuted and scaled encoding2 (None if no match)
    """
    if encoding1.shape != encoding2.shape:
        if verbose:
            print("❌ Encodings have different shapes")
        return False, None, None, None, None

    # First try direct match
    is_match: bool
    permuted: np.ndarray | None
    permuted2: np.ndarray | None
    is_match, permuted, perm, permuted2 = find_encoding_match(
        encoding1, encoding2, verbose=verbose
    )
    if is_match:
        if verbose:
            print("✅ Encodings match directly (no scaling needed)")
        assert np.allclose(permuted, permuted2, rtol=1e-9)  # type: ignore
        return True, 1.0, perm, permuted, permuted2

    # First try direct match with -1 scaling
    is_match, permuted, perm, permuted2 = find_encoding_match(
        encoding1, -encoding2, verbose=verbose
    )
    if is_match:
        if verbose:
            print("✅ Encodings match directly (with -1 scaling)")
        assert np.allclose(permuted, permuted2, rtol=1e-9)  # type: ignore
        return True, -1.0, perm, permuted, permuted2

    # If no direct match, try scaling
    max_abs1 = np.max(np.abs(encoding1))
    max_abs2 = np.max(np.abs(encoding2))

    if max_abs1 == 0 or max_abs2 == 0:
        if verbose:
            print("❌ One of the encodings is all zeros")
        return False, None, None, None, None

    scaling_factor = max_abs2 / max_abs1
    scaled_encoding1 = encoding1 * scaling_factor

    if verbose:
        print(f"\nTrying scaling factor: {scaling_factor:.4e}")
        print(f"Original max values: {max_abs1:.4e} vs {max_abs2:.4e}")
        print(
            f"After scaling: {np.max(np.abs(scaled_encoding1)):.4e} vs {max_abs2:.4e}"
        )

    # Check if scaled versions match
    is_match, permuted, perm, permuted2 = find_encoding_match(
        scaled_encoding1, encoding2, verbose=verbose
    )

    if is_match:
        if verbose:
            print(f"✅ Found match after scaling by {scaling_factor:.4e}")
        assert np.allclose(permuted, permuted2, rtol=1e-9)  # type: ignore
        return True, scaling_factor, perm, permuted, permuted2

    # If still no match, try with normalized versions
    normalized1 = encoding1 / max_abs1
    normalized2 = encoding2 / max_abs2

    if verbose:
        print("\nTrying with normalized encodings (divided by max abs value)")

    is_match, permuted, perm, permuted2 = find_encoding_match(
        normalized1, normalized2, verbose=verbose
    )

    if is_match:
        if verbose:
            print("✅ Found match after normalization")
        assert np.allclose(permuted, permuted2, rtol=1e-9)  # type: ignore
        return True, max_abs2 / max_abs1, perm, permuted, permuted2

    # If we get here, the encodings are truly different
    if verbose:
        print(
            "\n❌ Encodings are different even after trying scaling and normalization"
        )
        print("Statistics for diagnosis:")
        print(
            f"Encoding 1 - min: {np.min(encoding1):.4e}, max: {np.max(encoding1):.4e}, mean: {np.mean(encoding1):.4e}"
        )
        print(
            f"Encoding 2 - min: {np.min(encoding2):.4e}, max: {np.max(encoding2):.4e}, mean: {np.mean(encoding2):.4e}"
        )
        print(f"Ratio of max values (E2/E1): {max_abs2/max_abs1:.4e}")
        print(
            f"Ratio of min values (E2/E1): {np.min(np.abs(encoding2))/np.min(np.abs(encoding1)):.4e}"
        )

    return False, None, None, None, None
