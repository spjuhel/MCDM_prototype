import numpy as np
import pandas as pd

# Weight calculation: Scale to Weights
def weight_from_scale(criteria_scale, scale_range=(1, 5)):
    """
    Convert a dictionary of criteria scores into normalized weights that sum to 1.

    Parameters:
        criteria_scale (dict): Dictionary where keys are criteria names and values are importance scores.
        scale_range (tuple): Tuple specifying the minimum and maximum scale values (default is (1, 5)).

    Returns:
        dict: Dictionary with criteria names as keys and their normalized weights as values.

    Raises:
        ValueError: If the scale range is invalid, or scores are outside the range, or the total importance is zero.

    Example:
        >>> criteria_scale = {"Cost": 3, "Durability": 5, "Design": 2}
        >>> weight_from_scale(criteria_scale, scale_range=(1, 5))
        {'Cost': 0.3, 'Durability': 0.5, 'Design': 0.2}
    """

    min_scale, max_scale = scale_range
    if min_scale >= max_scale:
        raise ValueError("The minimum of the scale range must be less than the maximum.")
    if any(imp < min_scale or imp > max_scale for imp in criteria_scale.values()):
        raise ValueError(f"All importance scores must be within the range {scale_range}.")
    total_importance = sum(criteria_scale.values())
    if total_importance == 0:
        raise ValueError("The total importance value cannot be zero.")
    return {crit: imp / total_importance for crit, imp in criteria_scale.items()}

def weight_from_ahp(alternatives, pairwise_matrix, return_matrix=False):
    """
    Calculates weights using the Analytic Hierarchy Process (AHP) and detects logical inconsistencies.

    Parameters:
        alternatives (list): List of alternative names (str) to be evaluated.
        pairwise_matrix (np.ndarray): Pairwise comparison matrix where each element represents the relative importance.
        return_matrix (bool): If True, also returns the pairwise matrix and its DataFrame representation.

    Returns:
        dict: Calculated weights with alternatives as keys.
        (optional) pd.DataFrame: Pairwise matrix as a DataFrame if `return_matrix=True`.
        (optional) np.ndarray: Raw pairwise matrix if `return_matrix=True`.

    Raises:
        ValueError: If the pairwise matrix is missing, not square, or has inconsistent reciprocal relationships.

    Example:
        >>> pairwise_matrix = np.array([
                [1, 3, 0.5],
                [1/3, 1, 2],
                [2, 0.5, 1]
            ])
        >>> alternatives = ["Cost", "Durability", "Design"]
        >>> weight_from_ahp(alternatives, pairwise_matrix)
        {'Cost': 0.4444, 'Durability': 0.2963, 'Design': 0.2593}
    """
    if pairwise_matrix is None:
        raise ValueError(
            "pairwise_matrix is required. Use the `manual_input_weights` function to create it."
        )

    n = len(alternatives)
    if pairwise_matrix.shape != (n, n):
        raise ValueError(
            f"pairwise_matrix must be a square matrix of size {n}x{n}, corresponding to the number of alternatives provided."
        )

    # Validate pairwise matrix
    for i in range(n):
        if pairwise_matrix[i, i] != 1:
            raise ValueError(f"Diagonal elements must be 1 (self-comparison) for '{alternatives[i]}'.")

        for j in range(i + 1, n):
            if not np.isclose(pairwise_matrix[i, j] * pairwise_matrix[j, i], 1.0):
                raise ValueError(
                    f"Inconsistent reciprocal relationship detected: "
                    f"{alternatives[i]} > {alternatives[j]} = {pairwise_matrix[i, j]}, but "
                    f"{alternatives[j]} > {alternatives[i]} = {pairwise_matrix[j, i]}."
                )

    # Detect logical inconsistencies
    cyclic_inconsistencies = __find_cyclic_inconsistencies(pairwise_matrix, alternatives)

    if cyclic_inconsistencies:
        print("\nExamples of logical inconsistencies detected:")
        for idx, explanation in enumerate(cyclic_inconsistencies[:2]):  # Show up to 2 examples
            print(f"{idx + 1}. {explanation}")
        print("\nSuggestions:")
        print("- Revise and ensure pairwise comparisons are consistent across all alternatives.")
        print("- Use smaller scales (e.g., 1-3) to reduce inconsistencies.")

    # Calculate weights
    weights = __normalize_and_calculate_weights(pairwise_matrix, alternatives)

    if return_matrix:
        df_matrix = pd.DataFrame(pairwise_matrix, columns=alternatives, index=alternatives)
        return weights, df_matrix, pairwise_matrix

    return weights


def __find_cyclic_inconsistencies(pairwise_matrix, items):
    """
    Identify cyclic inconsistencies in the pairwise comparison matrix.

    Parameters:
        pairwise_matrix (np.ndarray): Pairwise comparison matrix.
        items (list): List of item names corresponding to the rows/columns of the matrix.

    Returns:
        list: List of strings describing detected cyclic inconsistencies.

    Example:
        >>> pairwise_matrix = np.array([
                [1, 3, 0.5],
                [1/3, 1, 2],
                [2, 0.5, 1]
            ])
        >>> items = ["Cost", "Durability", "Design"]
        >>> __find_cyclic_inconsistencies(pairwise_matrix, items)
        ["'Cost > Durability', 'Durability > Design', but 'Design > Cost'"]
    """
    n = len(items)
    inconsistencies = []

    for i in range(n):
        for j in range(n):
            for k in range(n):
                # Check for cyclic inconsistencies: A > B, B > C, but C > A
                if (
                    pairwise_matrix[i, j] > 1 and
                    pairwise_matrix[j, k] > 1 and
                    pairwise_matrix[k, i] > 1
                ):
                    inconsistencies.append(
                        f"'{items[i]} > {items[j]}', '{items[j]} > {items[k]}', but '{items[k]} > {items[i]}'"
                    )
    return inconsistencies


# Internal Functions for AHP Calculations
def __normalize_and_calculate_weights(pairwise_matrix, items):
    """
    Normalize the pairwise comparison matrix and calculate weights as the mean of each row.

    Parameters:
        pairwise_matrix (np.ndarray): Pairwise comparison matrix.
        items (list): List of item names corresponding to the rows/columns of the matrix.

    Returns:
        dict: Dictionary with item names as keys and their calculated weights as values.

    Example:
        >>> pairwise_matrix = np.array([
                [1, 3, 0.5],
                [1/3, 1, 2],
                [2, 0.5, 1]
            ])
        >>> items = ["Cost", "Durability", "Design"]
        >>> __normalize_and_calculate_weights(pairwise_matrix, items)
        {'Cost': 0.4444, 'Durability': 0.2963, 'Design': 0.2593}
    """
    column_sums = pairwise_matrix.sum(axis=0)
    normalized_matrix = pairwise_matrix / column_sums
    weights = normalized_matrix.mean(axis=1)
    return {item: round(weight, 4) for item, weight in zip(items, weights)}



# Weight calculation: From Categories to Criteria
def weight_from_categories(cat_crit, category_weights):
    """
    Distributes category weights to criteria based on their membership.

    Parameters:
        cat_crit (dict): Dictionary mapping categories to lists of criteria.
        category_weights (dict): Dictionary mapping categories to their weights.

    Returns:
        dict: Dictionary of criteria with their calculated weights.

    Raises:
        ValueError: If categories in `category_weights` are missing from `cat_crit`, or if weights do not sum to 1.

    Example:
        >>> cat_crit = {
                "Performance": ["Speed", "Acceleration"],
                "Comfort": ["Seating", "Suspension"],
                "Safety": ["Brakes"]
            }
        >>> category_weights = {"Performance": 0.5, "Comfort": 0.3, "Safety": 0.2}
        >>> weight_from_categories(cat_crit, category_weights)
        {'Speed': 0.25, 'Acceleration': 0.25, 'Seating': 0.15, 'Suspension': 0.15, 'Brakes': 0.2}
    """
    missing_categories = [cat for cat in category_weights if cat not in cat_crit]
    if missing_categories:
        raise ValueError(f"Categories in category_weights not found in cat_crit: {missing_categories}")
    total_weight = sum(category_weights.values())
    if not np.isclose(total_weight, 1.0):
        raise ValueError(f"Category weights must sum to 1.0. Current total: {total_weight:.4f}")
    criteria_weights = {}
    for category, weight in category_weights.items():
        criteria = cat_crit.get(category, [])
        num_criteria = len(criteria)
        if num_criteria == 0:
            raise ValueError(f"Category '{category}' has no criteria associated with it.")
        for crit in criteria:
            criteria_weights[crit] = weight / num_criteria
    return criteria_weights


# Manual Input Function
def manual_input_weights(items, method="pairwise", **helper_args):
    """
    Provides an interactive method for weight determination based on pairwise comparisons, scale scoring, or categories.

    Parameters:
        items (list): List of items (str) for which weights are being determined.
        method (str): Weighting method ("pairwise", "scale", or "categories").
        **helper_args: Additional arguments for specific methods:
            - For "pairwise": `max_range` (int) - maximum comparison value.
            - For "scale": `scale_range` (tuple) - scoring range.
            - For "categories": `cat_crit` (dict) - mapping of categories to criteria.

    Returns:
        dict: Calculated weights for the items.

    Raises:
        ValueError: If invalid inputs are detected or required arguments are missing.

    Example (Pairwise):
        >>> items = ["Cost", "Durability", "Design"]
        Enter pairwise comparisons (1 to 9 or fraction for reciprocal):
        How much more important is 'Cost' compared to 'Durability'? 3
        How much more important is 'Cost' compared to 'Design'? 1/2
        How much more important is 'Durability' compared to 'Design'? 2
        {'Cost': 0.4444, 'Durability': 0.2963, 'Design': 0.2593}

    Example (Scale):
        >>> items = ["Cost", "Durability", "Design"]
        Enter scores for each item (range 1 to 5):
        Score for 'Cost': 3
        Score for 'Durability': 5
        Score for 'Design': 2
        {'Cost': 0.3, 'Durability': 0.5, 'Design': 0.2}

    Example (Categories):
        >>> items = ["Speed", "Acceleration", "Seating", "Suspension", "Brakes"]
        >>> cat_crit = {
                "Performance": ["Speed", "Acceleration"],
                "Comfort": ["Seating", "Suspension"],
                "Safety": ["Brakes"]
            }
        Enter weights for each category (values must sum to 1):
        Weight for 'Performance': 0.5
        Weight for 'Comfort': 0.3
        Weight for 'Safety': 0.2
        {'Speed': 0.25, 'Acceleration': 0.25, 'Seating': 0.15, 'Suspension': 0.15, 'Brakes': 0.2}
    """
    if method == "pairwise":
        max_range = helper_args.get("max_range", 9)
        print("Enter pairwise comparisons (1 to {} or fraction for reciprocal):".format(max_range))
        pairwise_matrix = np.ones((len(items), len(items)))
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                while True:
                    try:
                        value = input(f"How much more important is '{items[i]}' compared to '{items[j]}'? ")
                        if "/" in value:
                            numerator, denominator = map(int, value.split("/"))
                            value = numerator / denominator
                        else:
                            value = float(value)
                        if 1 <= value <= max_range or (0 < value < 1):
                            pairwise_matrix[i, j] = value
                            pairwise_matrix[j, i] = 1 / value
                            break
                        else:
                            print(f"Enter a value between 1 and {max_range} or a valid fraction.")
                    except (ValueError, ZeroDivisionError):
                        print("Invalid input. Please try again.")
        return weight_from_ahp(items, pairwise_matrix=pairwise_matrix)
    elif method == "scale":
        scale_range = helper_args.get("scale_range", (1, 5))
        print(f"Enter scores for each item (range {scale_range[0]} to {scale_range[1]}):")
        scores = {}
        for item in items:
            while True:
                try:
                    score = float(input(f"Score for '{item}': "))
                    if scale_range[0] <= score <= scale_range[1]:
                        scores[item] = score
                        break
                    else:
                        print(f"Please enter a value between {scale_range[0]} and {scale_range[1]}.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        return weight_from_scale(scores, scale_range)
    elif method == "categories":
        cat_crit = helper_args.get("cat_crit")
        if not isinstance(cat_crit, dict):
            raise ValueError("For 'categories' method, 'cat_crit' (dict) must be provided in helper_args.")
        print("Enter weights for each category (values must sum to 1):")
        category_weights = {}
        for category in cat_crit:
            while True:
                try:
                    weight = float(input(f"Weight for '{category}': "))
                    if 0 <= weight <= 1:
                        category_weights[category] = weight
                        break
                    else:
                        print("Please enter a value between 0 and 1.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        total_weight = sum(category_weights.values())
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Category weights must sum to 1. Current total: {total_weight:.4f}")
        return weight_from_categories(cat_crit, category_weights)
    else:
        raise ValueError(f"Invalid method '{method}'. Choose from 'pairwise', 'scale', or 'categories'.")
