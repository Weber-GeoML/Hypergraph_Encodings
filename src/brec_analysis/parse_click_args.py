"""Parse click arguments for main"""

from brec_analysis.categories_to_check import VALID_CATEGORIES
from brec_analysis.encodings_to_check import ENCODINGS_TO_CHECK


def parse_encoding(encodings: str) -> list[tuple[str, str]] | None:
    """Parse encoding indices"""
    # Parse encoding indices
    try:
        if "-" in encodings:
            # Handle range format (e.g., "0-3")
            start, end = map(int, encodings.split("-"))
            encoding_indices = list(range(start, end + 1))
            print(f"start: {start}, end: {end}, encoding_indices: {encoding_indices}")
        else:
            # Handle comma-separated format (e.g., "0,2,3")
            encoding_indices = [int(i) for i in encodings.split(",")]

        # Validate indices
        max_idx = len(ENCODINGS_TO_CHECK) - 1
        invalid_indices = [i for i in encoding_indices if i < 0 or i > max_idx]
        if invalid_indices:
            raise ValueError(
                f"Invalid encoding indices: {invalid_indices}. "
                f"Must be between 0 and {max_idx}"
            )

        # Select specified encodings
        selected_encodings = [ENCODINGS_TO_CHECK[i] for i in encoding_indices]
        return selected_encodings
    except ValueError as e:
        print(f"Error parsing encoding indices: {e}")
        print("\nAvailable encodings:")
        for i, (code, name) in enumerate(ENCODINGS_TO_CHECK):
            print(f"{i}: {code} ({name})")
        return None


def parse_categories(categories_str: str) -> list[str]:
    """Parse category indices from string input.

    Args:
        categories_str: String containing category indices (e.g., "0" or "0,3" or "0-3")

    Returns:
        List of selected category names

    Raises:
        ValueError: If invalid indices are provided
    """
    try:
        # Handle range format (e.g., "0-3")
        if "-" in categories_str:
            start, end = map(int, categories_str.split("-"))
            category_indices = list(range(start, end + 1))
        else:
            # Handle comma-separated format (e.g., "0,2,3")
            category_indices = [int(i) for i in categories_str.split(",")]

        # Validate indices
        max_idx = len(VALID_CATEGORIES) - 1
        invalid_indices = [i for i in category_indices if i < 0 or i > max_idx]
        if invalid_indices:
            raise ValueError(
                f"Invalid category indices: {invalid_indices}\n"
                f"Must be between 0 and {max_idx}"
            )

        # Get selected categories
        selected_categories = [VALID_CATEGORIES[i][0] for i in category_indices]

        # Print selection info
        print("\nSelected categories:")
        for idx in category_indices:
            code, range_, name = VALID_CATEGORIES[idx]
            print(f"  {name} (range: {range_[0]}-{range_[1]})")

        return selected_categories

    except ValueError as e:
        print(f"Error parsing category indices: {e}")
        print("\nAvailable categories:")
        for i, (code, range_, name) in enumerate(VALID_CATEGORIES):
            print(f"{i}: {name} (range: {range_[0]}-{range_[1]})")
        raise
