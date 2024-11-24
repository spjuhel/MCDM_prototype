import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def sort_metrics_df(metrics_df, alt_cols, group_cols, unc_cols):
    """Sort metrics_df based on alt_cols, group_cols, and unc_cols."""
    return metrics_df.sort_values(by=alt_cols + group_cols + unc_cols)


def create_alternatives_df(metrics_df, alt_cols, alternative_colors=None):
    """Create a DataFrame for unique alternatives and populate alternative colors."""
    if not alt_cols:
        raise ValueError("No alternative column given.")

    alternatives_df = metrics_df[alt_cols].drop_duplicates()
    alternatives_df.insert(0, 'Alternative ID', ['A' + str(idx) for idx in range(1, len(alternatives_df) + 1)])

    alt_colors = {}
    if alternative_colors:
        missing_colors = set(alternatives_df['Alternative ID']) - set(alternative_colors.keys())
        if missing_colors:
            raise ValueError(f"Missing colors for the following alternatives: {missing_colors}")
        alt_colors = {alt_id: alternative_colors[alt_id] for alt_id in alternatives_df['Alternative ID']}
    else:
        default_colors = plt.cm.tab10(np.linspace(0, 1, len(alternatives_df)))
        for idx, alt_id in enumerate(alternatives_df['Alternative ID']):
            alt_colors[alt_id] = np.array(default_colors[idx][:3])

    return alternatives_df, alt_colors


def create_unc_samples_df(metrics_df, unc_cols):
    """Create a DataFrame for unique samples from uncertainty columns."""
    if unc_cols:
        unc_smpls_df = metrics_df[unc_cols].drop_duplicates().reset_index(drop=True)
        unc_smpls_df.insert(0, 'Sample ID', ['S' + str(idx) for idx in range(1, len(unc_smpls_df) + 1)])
        return unc_smpls_df
    return None


def assign_criteria_weights(weights, crit_cols):
    """Assign weights to criteria, defaulting to equal weights if not provided."""
    if not weights:
        return {crit: round(1 / len(crit_cols), 2) for crit in crit_cols}
    return weights


def validate_duplicates(metrics_df, alt_cols, group_cols, unc_cols):
    """Check for duplicate rows in metrics_df."""
    if metrics_df.duplicated(subset=alt_cols + group_cols + unc_cols, keep=False).any():
        raise ValueError("Duplicated rows of alt_cols, group_cols, and sample detected.")


def create_criteria_df(crit_cols, weights, objectives):
    """Create a DataFrame for criteria with weights and objectives."""
    return pd.DataFrame(
        [{'Criteria ID': f'C{idx + 1}', 'Criteria': crit, 'Weight': weights[crit], 'Objective': objectives[crit]}
         for idx, crit in enumerate(crit_cols)]
    )


def create_categorized_criteria_df(crit_cats, crit_cols, crit_df):
    """Create a categorized criteria DataFrame and return updated crit_cats."""
    # Ensure crit_cats includes all criteria
    if not crit_cats:
        crit_cats = {crit: [crit] for crit in crit_cols}
    else:
        # Fill in missing criteria in crit_cats
        missing_criteria = set(crit_cols) - set(crit_cats.keys())
        for crit in missing_criteria:
            crit_cats[crit] = [crit]

    data = [{'Cat ID': f'CAT{idx + 1}', 'Category': category, 'Criteria': crit}
            for idx, (category, crit_list) in enumerate(crit_cats.items())
            for crit in crit_list]
    cat_crit_df = pd.DataFrame(data)

    # Return both the DataFrame and updated crit_cats dictionary
    return cat_crit_df.merge(crit_df, on='Criteria'), crit_cats


def create_group_weights(metrics_df, group_cols, group_weights):
    """Create group weights and group DataFrame."""
    if not group_cols:
        return None, {}

    # Extract unique groups
    groups_df = metrics_df[group_cols].drop_duplicates().reset_index(drop=True)
    groups_df.insert(0, 'Group ID', ['G' + str(idx + 1) for idx in range(len(groups_df))])

    # Ensure `_merge` is not present in the final `groups_df`
    groups_df = groups_df.drop(columns=['_merge'], errors='ignore')

    # Calculate weights for the groups if provided
    updated_group_weights = {}
    for group_col in group_cols:
        temp_df = pd.DataFrame(metrics_df[group_col].drop_duplicates(), columns=[group_col])
        temp_df['Weight'] = np.nan

        if group_col in group_weights:
            temp_df['Weight'] = temp_df[group_col].map(group_weights[group_col]).fillna(np.nan)

        sum_defined_weights = temp_df['Weight'].sum(skipna=True)
        remaining_members = temp_df['Weight'].isna().sum()

        if remaining_members > 0:
            equal_weight = (1 - sum_defined_weights) / remaining_members
            temp_df['Weight'] = temp_df['Weight'].fillna(equal_weight)

        updated_group_weights[group_col] = temp_df

    return groups_df, updated_group_weights




def merge_components_to_dm_df(alternatives_df, groups_df, unc_smpls_df, metrics_df, alt_cols, group_cols, unc_cols, crit_cols):
    """Merge alternatives, groups, and uncertainty samples into the final decision-making DataFrame."""
    dm_df = alternatives_df.copy()
    dm_df['_merge'] = 1  # Temporary column for merging

    # Merge groups_df if it exists
    if isinstance(groups_df, pd.DataFrame):
        groups_df['_merge'] = 1
        dm_df = dm_df.merge(groups_df, on='_merge', how='left')  # Use 'left' join
        groups_df.drop('_merge', axis=1, errors='ignore', inplace=True)  # Clean up _merge in groups_df

    # Merge unc_smpls_df if it exists
    if isinstance(unc_smpls_df, pd.DataFrame):
        unc_smpls_df['_merge'] = 1
        dm_df = dm_df.merge(unc_smpls_df, on='_merge', how='left')  # Use 'left' join
        unc_smpls_df.drop('_merge', axis=1, errors='ignore', inplace=True)  # Clean up _merge in unc_smpls_df

    # Clean up _merge in dm_df
    dm_df.drop('_merge', axis=1, errors='ignore', inplace=True)

    # Final merge with metrics_df
    dm_df = pd.merge(
        dm_df,
        metrics_df[alt_cols + group_cols + unc_cols + crit_cols],
        on=alt_cols + group_cols + unc_cols,
        how='left'
    )

    return dm_df



