
from typing import Dict, List, Optional
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
import matplotlib.lines as mlines
import matplotlib.cm as cm
import seaborn as sns


# Importing various methods from pyrepo_mcda
from pyrepo_mcda.mcda_methods import AHP, ARAS, COCOSO, CODAS, COPRAS, CRADIS, EDAS, MABAC, MARCOS, MULTIMOORA, MULTIMOORA_RS, PROMETHEE_II, PROSA_C, SAW, SPOTIS, TOPSIS, VIKOR, VMCM, WASPAS, VIKOR_SMAA
from pyrepo_mcda.compromise_rankings import copeland, dominance_directed_graph, rank_position_method
from pyrepo_mcda import distance_metrics as dists
from pyrepo_mcda import normalizations as norms

from pyrepo_mcda import correlations as corrs
from pyrepo_mcda import weighting_methods as mcda_weights

# Importing additional functions from pyrepo_mcda
from pyrepo_mcda.additions import rank_preferences

# Utility functions

#%% Helper functions for the calc_rankings method

def prepare_matrix(sg_df, crit_cols):
    """
    Prepare the decision matrix by filtering NaN alternatives and adjusting the matrix to handle non-positive values.

    Parameters:
    - sg_df (pd.DataFrame): Subset of the decision matrix for a specific group-scenario combination.
    - crit_cols (list): List of criteria columns.

    Returns:
    - matrix (np.ndarray): Adjusted decision matrix as a numpy array.
    - nan_alt_rows (pd.Series): Boolean Series indicating rows with NaN values.
    - sg_df_filtered (pd.DataFrame): Filtered DataFrame without NaN values in criteria columns.
    """
    # Identify rows with NaN values in the criteria columns
    nan_alt_rows = sg_df[crit_cols].isna().any(axis=1)

    # Remove rows with NaN values
    sg_df_filtered = sg_df[~nan_alt_rows]

    # Adjust criteria values to avoid issues with zero or negative values
    matrix_df = sg_df_filtered[crit_cols].copy()
    matrix_df = matrix_df.apply(lambda col: col + abs(col.min()) + 1)

    # Add a small random positive value to the matrix to avoid numerical issues
    matrix = matrix_df.to_numpy() + np.random.rand(*matrix_df.shape) * 1e-4

    return matrix, nan_alt_rows, sg_df_filtered

def apply_constraints(sg_df, constraints, derived_columns, base_cols):
    """
    Apply constraints to filter the decision matrix for a specific group-scenario combination.

    Parameters:
    - sg_df (pd.DataFrame): Subset of the decision matrix for a specific group-scenario combination.
    - constraints (dict): Dictionary of constraints to apply to the decision matrix.
    - derived_columns (list): List of derived columns to include in the output DataFrame.
    - base_cols (list): List of base columns to retain in the output DataFrame.

    Returns:
    - sg_df_filtered (pd.DataFrame): Filtered DataFrame satisfying the constraints.
    - alt_exc_const_df (pd.DataFrame): DataFrame of alternatives excluded by constraints.
    """
    sg_df_filtered, boolean_df = filter_dataframe(
        sg_df,
        filter_conditions=constraints,
        derived_columns=derived_columns,
        base_cols=base_cols,
    )
    alt_exc_const_df = boolean_df[~(boolean_df[constraints.keys()] == True).all(axis=1)]

    return sg_df_filtered, alt_exc_const_df

# Calculate rankings using MCDM methods and compromise rankings
def calculate_rankings(matrix, weights, types, mcdm_methods, comp_ranks, temp_ranks_df):
    """
    Calculate rankings using MCDM methods and compromise rankings.

    Parameters:
    - matrix (np.ndarray): Decision matrix for criteria.
    - weights (np.ndarray): Criteria weights.
    - types (np.ndarray): Criteria objectives.
    - mcdm_methods (Dict): MCDM methods to apply.
    - comp_ranks (Dict): Compromise ranking methods to apply.
    - temp_ranks_df (pd.DataFrame): Temporary DataFrame to store rankings.

    Returns:
    - pd.DataFrame: Updated rankings DataFrame with MCDM and compromise rankings.
    """
    # Perform MCDM calculations
    for method_name, method in mcdm_methods.items():
        if not isinstance(method, SPOTIS):
            pref = method(matrix, weights, types)
        else:
            bounds_min = np.amin(matrix, axis=0)
            bounds_max = np.amax(matrix, axis=0)
            bounds = np.vstack((bounds_min, bounds_max))
            pref = method(matrix, weights, types, bounds)

        if isinstance(method, MULTIMOORA):
            temp_ranks_df[method_name] = method(matrix, weights, types)
        elif isinstance(method, (VIKOR, SPOTIS)):
            temp_ranks_df[method_name] = rank_preferences(pref, reverse=False)
        else:
            temp_ranks_df[method_name] = rank_preferences(pref, reverse=True)

    # Perform compromise rankings
    for comp_name, comp_func in comp_ranks.items():
        temp_ranks_df[comp_name] = comp_func(temp_ranks_df[mcdm_methods.keys()].to_numpy())

    return temp_ranks_df


# Rank the specified columns in a DataFrame according to the provided ranking objectives
def ranks_columns(df, columns, objectives):
    """
    Rank specified columns in a DataFrame according to provided ranking objectives.

    Parameters:
        df (pandas.DataFrame): 
            The DataFrame containing the data to be ranked.
        columns (list of str): 
            A list of column names to be ranked.
        objectives (dict of {str: callable}): 
            A dictionary with column names as keys and ranking objective functions as values.

    Returns:
        pandas.DataFrame: A new DataFrame with the specified columns ranked according to the objectives.
    """
        
    # Mapping for function selection
    FUNCTION_MAP = {'1': False, '-1': True}
    
    # Copy the input DataFrame
    ranked_df = df.copy()

    # Iterate over the columns to be ranked
    for col in columns:
        # Rank the column based on the specified objective
        ranked_df[col] = ranked_df[col].rank(method='min', ascending=FUNCTION_MAP[str(objectives[col])])
        # Convert the ranks to integers
        ranked_df[col] = ranked_df[col].astype(int)
        
    return ranked_df