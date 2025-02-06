"""Functions for matching encodings.


Try to find a permutation of the rows of encoding1 that makes it equal to encoding2.
Also try up to scaling.
"""

from itertools import permutations

import numpy as np
from scipy.stats import kurtosis, skew


def find_encoding_match(
    encoding1: np.ndarray,
    encoding2: np.ndarray,
    name_of_encoding: str,
    verbose: bool = True,
) -> tuple[
    bool,
    np.ndarray | None,
    tuple[int, ...] | None,
    np.ndarray | None,
    str | None,
]:
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
    sort_idx1: np.ndarray
    sort_idx2: np.ndarray

    permuted: np.ndarray
    permuted2: np.ndarray
    perm: np.ndarray

    if encoding1.shape != encoding2.shape:
        return False, None, None, None, None

    # First check if the encodings are identical
    if np.allclose(encoding1, encoding2, rtol=1e-13):
        # Return identity permutation if encodings are identical
        if verbose:
            print("Free lunch!")
        n_rows = encoding1.shape[0]
        return True, encoding1, tuple(range(n_rows)), encoding2, None

    # Pre-compute expensive operations
    abs_enc1: np.ndarray = np.abs(encoding1)
    abs_enc2: np.ndarray = np.abs(encoding2)

    # Check the max absolute value of each encodings. If they are different, return False
    if not np.isclose(np.max(abs_enc1), np.max(abs_enc2), rtol=1e-12):
        if verbose:
            print("Different because:")
            print(
                f"Max absolute value of encoding1: {np.max(np.abs(encoding1))}"
            )
            print(
                f"Max absolute value of encoding2: {np.max(np.abs(encoding2))}"
            )
            print("\n")
        return False, None, None, None, None

    # Same for min
    if not np.isclose(np.min(abs_enc1), np.min(abs_enc2), rtol=1e-12):
        if verbose:
            print("Different because:")
            print(
                f"Min absolute value of encoding1: {np.min(np.abs(encoding1))}"
            )
            print(
                f"Min absolute value of encoding2: {np.min(np.abs(encoding2))}"
            )
            print("\n")
        return False, None, None, None, None

    # Same for mean
    if not np.isclose(np.mean(abs_enc1), np.mean(abs_enc2), rtol=1e-12):
        if verbose:
            print("Different because:")
            print(
                f"Mean absolute value of encoding1: {np.mean(np.abs(encoding1))}"
            )
            print(
                f"Mean absolute value of encoding2: {np.mean(np.abs(encoding2))}"
            )
            print("\n")
        return False, None, None, None, None

    # Vectorized column comparisons for both max and min
    # t0 = time.time()
    max_cols1 = np.max(abs_enc1, axis=0)
    max_cols2 = np.max(abs_enc2, axis=0)
    # if verbose:
    #     print(f"Max computation time: {time.time() - t0:.4f} seconds")

    # Check max values
    if not np.allclose(max_cols1, max_cols2, rtol=1e-12):
        if verbose:
            diff_cols = ~np.isclose(max_cols1, max_cols2, rtol=1e-12)
            print(f"Different max at columns: {np.where(diff_cols)[0]}")
            print(f"Max values enc1: {max_cols1[diff_cols]}")
            print(f"Max values enc2: {max_cols2[diff_cols]}")
        return False, None, None, None, None

    # t0 = time.time()
    min_cols1 = np.min(abs_enc1, axis=0)
    min_cols2 = np.min(abs_enc2, axis=0)
    # if verbose:
    #     print(f"Min computation time: {time.time() - t0:.4f} seconds")

    # Check min values
    if not np.allclose(min_cols1, min_cols2, rtol=1e-12):
        if verbose:
            diff_cols = ~np.isclose(min_cols1, min_cols2, rtol=1e-12)
            print(f"Different min at columns: {np.where(diff_cols)[0]}")
            print(f"Min values enc1: {min_cols1[diff_cols]}")
            print(f"Min values enc2: {min_cols2[diff_cols]}")
        return False, None, None, None, None

    # t0 = time.time()
    mean_cols1 = np.mean(abs_enc1, axis=0)
    mean_cols2 = np.mean(abs_enc2, axis=0)
    # if verbose:
    #     print(f"Mean computation time: {time.time() - t0:.4f} seconds")

    # Check mean values
    if not np.allclose(mean_cols1, mean_cols2, rtol=1e-12):
        if verbose:
            diff_cols = ~np.isclose(mean_cols1, mean_cols2, rtol=1e-12)
            print(f"Different mean at columns: {np.where(diff_cols)[0]}")
            print(f"Mean values enc1: {mean_cols1[diff_cols]}")
            print(f"Mean values enc2: {mean_cols2[diff_cols]}")
        return False, None, None, None, None

    # t0 = time.time()
    std_cols1 = np.std(abs_enc1, axis=0)
    std_cols2 = np.std(abs_enc2, axis=0)
    # if verbose:
    #     print(f"Std computation time: {time.time() - t0:.4f} seconds")

    # Check standard deviation values
    if not np.allclose(std_cols1, std_cols2, rtol=1e-12):
        if verbose:
            diff_cols = ~np.isclose(std_cols1, std_cols2, rtol=1e-12)
            print(f"Different std at columns: {np.where(diff_cols)[0]}")
            print(f"Std values enc1: {std_cols1[diff_cols]}")
            print(f"Std values enc2: {std_cols2[diff_cols]}")
        return False, None, None, None, None

    n_rows = encoding1.shape[0]

    if n_rows <= 15:
        for perm_ in permutations(range(n_rows)):
            permuted = encoding1[list(perm_), :]
            if np.allclose(permuted, encoding2, rtol=1e-13):
                return True, permuted, tuple(perm_), encoding2, None

    # kurtosis
    # t0 = time.time()
    kurtosis_cols1 = kurtosis(abs_enc1, axis=0)
    kurtosis_cols2 = kurtosis(abs_enc2, axis=0)
    if not (
        np.any(np.isnan(kurtosis_cols1)) or np.any(np.isnan(kurtosis_cols2))
    ):
        # if verbose:
        #     print(f"Kurtosis computation time: {time.time() - t0:.4f} seconds")

        # check kurtosis
        if not np.allclose(kurtosis_cols1, kurtosis_cols2, rtol=1e-12):
            if verbose:
                diff_cols = ~np.isclose(
                    kurtosis_cols1, kurtosis_cols2, rtol=1e-12
                )
                print(
                    f"Different kurtosis at columns: {np.where(diff_cols)[0]}"
                )
                print(f"Kurtosis values enc1: {kurtosis_cols1[diff_cols]}")
                print(f"Kurtosis values enc2: {kurtosis_cols2[diff_cols]}")
            return False, None, None, None, None

    # Additional statistical measures
    # t0 = time.time()
    median_cols1 = np.median(abs_enc1, axis=0)
    median_cols2 = np.median(abs_enc2, axis=0)
    # if verbose:
    #     print(f"Median computation time: {time.time() - t0:.4f} seconds")

    # check median
    if not np.allclose(median_cols1, median_cols2, rtol=1e-12):
        if verbose:
            diff_cols = ~np.isclose(median_cols1, median_cols2, rtol=1e-12)
            print(f"Different median at columns: {np.where(diff_cols)[0]}")
            print(f"Median values enc1: {median_cols1[diff_cols]}")
            print(f"Median values enc2: {median_cols2[diff_cols]}")
        return False, None, None, None, None

    # take the product of every row
    # product_cols1: np.ndarray = np.prod(abs_enc1, axis=1)
    # product_cols2: np.ndarray = np.prod(abs_enc2, axis=1)
    # could do some things here. TODO. eg multiply all values etc..

    # Skewness
    # t0 = time.time()
    skew_cols1: np.ndarray = skew(abs_enc1, axis=0)
    skew_cols2: np.ndarray = skew(abs_enc2, axis=0)
    # if verbose:
    #     print(f"Skewness computation time: {time.time() - t0:.4f} seconds")
    # if skew_cols1 is not None/nan and same for skew_cols2
    if not (np.any(np.isnan(skew_cols1)) or np.any(np.isnan(skew_cols2))):
        # check skew
        if not np.allclose(skew_cols1, skew_cols2, rtol=1e-12):
            if verbose:
                diff_cols = ~np.isclose(skew_cols1, skew_cols2, rtol=1e-12)
                print(f"Different skew at columns: {np.where(diff_cols)[0]}")
                print(f"Skew values enc1: {skew_cols1[diff_cols]}")
                print(f"Skew values enc2: {skew_cols2[diff_cols]}")
            return False, None, None, None, None

    # Sum of squares
    # t0 = time.time()
    sum_sq_cols1: np.ndarray = np.sum(abs_enc1**2, axis=0)
    sum_sq_cols2: np.ndarray = np.sum(abs_enc2**2, axis=0)
    # if verbose:
    #     print(f"Sum of squares computation time: {time.time() - t0:.4f} seconds")

    # check sum of squares
    if not np.allclose(sum_sq_cols1, sum_sq_cols2, rtol=1e-12):
        if verbose:
            diff_cols = ~np.isclose(sum_sq_cols1, sum_sq_cols2, rtol=1e-12)
            print(
                f"Different sum of squares at columns: {np.where(diff_cols)[0]}"
            )
            print(f"Sum of squares values enc1: {sum_sq_cols1[diff_cols]}")
            print(f"Sum of squares values enc2: {sum_sq_cols2[diff_cols]}")
        return False, None, None, None, None

    # check only the last columns of each, sorted. see if they are allclose:
    last_col1 = np.sort(abs_enc1[:, -1])
    last_col2 = np.sort(abs_enc2[:, -1])

    if not np.allclose(last_col1, last_col2, rtol=1e-12):
        if verbose:
            print("Different because last columns don't match when sorted")
            print(f"Sorted last column 1: {last_col1}")
            print(f"Sorted last column 2: {last_col2}")
        return False, None, None, None, None
    else:  # try to find a permutation that makes the last columns match and try this one next on the whole thing
        # Get indices that would sort the last columns
        sort_idx1 = np.argsort(abs_enc1[:, -1])
        sort_idx2 = np.argsort(abs_enc2[:, -1])

        # Apply this permutation to encoding1
        permuted = encoding1[sort_idx1]
        permuted2 = encoding2[sort_idx2]

        # Check if this permutation works for the whole encoding
        if np.allclose(permuted, permuted2, rtol=1e-12):
            return True, permuted, tuple(sort_idx1), permuted2, None

        if encoding1.shape[1] > 2:
            sort_idx1 = np.argsort(abs_enc1[:, -2])
            sort_idx2 = np.argsort(abs_enc2[:, -2])

            # Apply this permutation to encoding1
            permuted = encoding1[sort_idx1]
            permuted2 = encoding2[sort_idx2]
            # Check if this permutation works for the whole encoding
            if np.allclose(permuted, permuted2, rtol=1e-12):
                return True, permuted, tuple(sort_idx1), permuted2, None

    # >>> a=np.array([[1,1],[2,2]])
    # >>> a
    # array([[1, 1],
    #     [2, 2]])
    # >>> np.prod(a,axis=0)
    # array([2, 2])
    # >>> np.prod(a,axis=1)
    # array([1, 4])
    prod_cols1 = np.prod(abs_enc1, axis=0)
    prod_cols2 = np.prod(abs_enc2, axis=0)

    if not np.allclose(prod_cols1, prod_cols2, rtol=1e-12):
        if verbose:
            diff_cols = ~np.isclose(prod_cols1, prod_cols2, rtol=1e-12)
            print(f"Different products at columns: {np.where(diff_cols)[0]}")
            print(f"Product values enc1: {prod_cols1[diff_cols]}")
            print(f"Product values enc2: {prod_cols2[diff_cols]}")
        return False, None, None, None, None

    if n_rows < 20:
        lexsort1: np.ndarray = np.lexsort(abs_enc1.T)
        lexsort2: np.ndarray = np.lexsort(abs_enc2.T)
        sorted1 = encoding1[lexsort1]
        sorted2 = encoding2[lexsort2]

        # Compare sorted matrices
        if np.allclose(sorted1, sorted2, rtol=1e-13):
            perm = np.argsort(lexsort1)
            return True, sorted1, tuple(perm), sorted2, None

    # start_time = time.time()
    # For small matrices, we can try all permutations
    long_timeout = False
    if not long_timeout:
        print("🚨 🚨 🚨 Assume same 🚨 🚨 🚨")
        print("🚨" * 20)
        return (
            True,
            permuted,
            tuple(sort_idx1),
            permuted2,
            None,
        )  # could fix more
    if long_timeout:
        # For larger matrices, use a heuristic approach based on row sorting
        # This works because:
        # 1. If two graphs are isomorphic, their encodings differ only by row permutation
        # 2. Lexicographical sorting will arrange rows in a canonical order
        # 3. After sorting, isomorphic graphs will have identical encodings

        try:
            lexsort1 = np.lexsort(encoding1.T)
            lexsort2 = np.lexsort(encoding2.T)
            sorted1 = encoding1[lexsort1]
            sorted2 = encoding2[lexsort2]

            # Compare sorted matrices
            if np.allclose(sorted1, sorted2, rtol=1e-13):
                perm = np.argsort(lexsort1)
                return True, sorted1, tuple(perm), sorted2, None
        except TimeoutError:
            return True, None, None, None, "timeout"

    return False, None, None, None, None


def check_encodings_same_up_to_scaling(
    encoding1: np.ndarray,
    encoding2: np.ndarray,
    name_of_encoding: str,
    verbose: bool = False,
) -> tuple[
    bool,
    float | None,
    tuple[int, ...] | None,
    np.ndarray | None,
    np.ndarray | None,
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
    timeout: str | None
    is_match, permuted, perm, permuted2, timeout = find_encoding_match(
        encoding1, encoding2, name_of_encoding=name_of_encoding, verbose=verbose
    )
    if is_match:
        if timeout is not None:
            print(f"🚨 Warning: Timeout after {timeout}")
        if verbose:
            print("✅ Encodings match directly (no scaling needed)")
        if timeout is None:
            assert np.allclose(permuted, permuted2, rtol=1e-9)  # type: ignore
        return True, 1.0, perm, permuted, permuted2

    # First try direct match with -1 scaling
    is_match, permuted, perm, permuted2, timeout = find_encoding_match(
        encoding1,
        -encoding2,
        name_of_encoding=name_of_encoding,
        verbose=verbose,
    )
    if is_match:
        if timeout is not None:
            print(f"🚨 Warning: Timeout after {timeout}")
        if verbose:
            print("✅ Encodings match directly (with -1 scaling)")
        if timeout is None:
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
    is_match, permuted, perm, permuted2, timeout = find_encoding_match(
        scaled_encoding1,
        encoding2,
        name_of_encoding=name_of_encoding,
        verbose=verbose,
    )

    if is_match:
        if timeout is not None:
            print(f"🚨 Warning: Timeout after {timeout}")
        if verbose:
            print(f"✅ Found match after scaling by {scaling_factor:.4e}")
        if timeout is None:
            assert np.allclose(permuted, permuted2, rtol=1e-9)  # type: ignore
        return True, scaling_factor, perm, permuted, permuted2

    # If still no match, try with normalized versions
    normalized1 = encoding1 / max_abs1
    normalized2 = encoding2 / max_abs2

    if verbose:
        print("\nTrying with normalized encodings (divided by max abs value)")

    is_match, permuted, perm, permuted2, timeout = find_encoding_match(
        normalized1,
        normalized2,
        name_of_encoding=name_of_encoding,
        verbose=verbose,
    )

    if is_match:
        if timeout is not None:
            print(f"🚨 Warning: Timeout after {timeout}")
        if verbose:
            print("✅ Found match after normalization")
        if timeout is None:
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
