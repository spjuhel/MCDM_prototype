from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import copy

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
import matplotlib.lines as mlines
import matplotlib.cm as cm
import seaborn as sns
import tabulate as tb


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
from .MCDMoutput_dev import RanksOutput
from .utils.other_utils import filter_dataframe

'''
ToDO:
    
    - Make the plot methods as for the CalcRank results
        - E.g., Distribution of criteria values
    - Make the calc conditional criteria value-at-risk
    - Create a color attribute for each alternative. A dictionary of colors. 

'''
# Define the MCDM ranking methods
MCDM_DEFAULT = {#'AHP': AHP(),
          'Topsis': TOPSIS(),
          'Saw': SAW(),
          'Vikor': VIKOR()
    }

# Define the compromised ranking function of the rank matrices
COMP_DEFAULT = {'copeland': copeland,
      }


class DecisionMatrix:
    def __init__(
        self,
        metrics_df: pd.DataFrame,
        objectives: Optional[Dict[str, int]] = None,
        alt_cols: Optional[List[str]] = None,
        crit_cols: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        group_cols: Optional[List[str]] = [],
        group_weights: Optional[Dict[str, float]] = None,
        unc_cols: Optional[List[str]] = [],
        crit_cats: Optional[Dict[str, List[str]]] = None,
        alternative_colors: Optional[Dict[str, np.ndarray]] = None,
    ):
        # Assign input parameters to class attributes as copies
        self.metrics_df = metrics_df.copy()
        self.group_cols = group_cols.copy() if group_cols is not None else []
        self.unc_cols = unc_cols.copy() if unc_cols is not None else []

        # Handle alt_cols
        if alt_cols is None:
            # Add a dummy 'Alternative' column with unique identifiers
            self.metrics_df.insert(0, 'Alternative', [f'Alternative {i+1}' for i in range(len(metrics_df))])
            self.alt_cols = ['Alternative']
        else:
            self.alt_cols = alt_cols.copy()

        # If crit_cols is not provided, infer it as all columns except alt_cols, group_cols, and unc_cols
        if crit_cols is None:
            excluded_cols = set(self.alt_cols + self.group_cols + self.unc_cols)
            self.crit_cols = [col for col in self.metrics_df.columns if col not in excluded_cols]
        else:
            self.crit_cols = crit_cols.copy()

        # Handle objectives
        if objectives is None:
            # If no objectives are provided, assign a default of 1 (maximize) to all criteria
            self.objectives = {crit: 1 for crit in self.crit_cols}
        else:
            self.objectives = copy.deepcopy(objectives)
            # Ensure all criteria have an objective; default to 1 if not provided
            for crit in self.crit_cols:
                if crit not in self.objectives:
                    self.objectives[crit] = 1

       # Define the custom rounding function for weights
        def custom_round(value):
            decimal_count = len(str(value).split(".")[1]) if "." in str(value) else 0
            decimals = 2 if decimal_count >= 2 else 1
            return round(value, decimals)

        # Calculate criteria weights if not provided or partially provided
        if weights:
            # Calculate the sum of explicitly defined weights
            defined_weight_sum = sum(weights.values())
            
            # Get the criteria without explicitly defined weights
            undefined_criteria = [crit for crit in self.crit_cols if crit not in weights]
            
            # Check if there are undefined criteria
            if undefined_criteria:
                # Distribute remaining weight equally among undefined criteria
                remaining_weight = 1 - defined_weight_sum
                if remaining_weight < 0:
                    raise ValueError("The sum of explicitly defined weights exceeds 1.")
                equal_weight = custom_round(remaining_weight / len(undefined_criteria))
                
                # Assign equal weights to undefined criteria
                for crit in undefined_criteria:
                    weights[crit] = equal_weight
        else:
            # If no weights are provided, assign equal weights to all criteria
            weights = {crit: custom_round(1 / len(self.crit_cols)) for crit in self.crit_cols}

        # Store the calculated weights
        self.weights = weights.copy()


        self.group_weights = group_weights.copy() if group_weights is not None else {}
        self.crit_cats = crit_cats.copy() if crit_cats is not None else {}


        # Initialize other attributes as None
        self.dm_df = None
        self.alternatives_df = None
        self.crit_df = None
        self.cat_crit_df = None
        self.groups_df = None
        self.unc_smpls_df = None

        # Sort the dm_df based on alt_cols, group_cols, and unc_cols
        sort_columns = self.alt_cols + self.group_cols + self.unc_cols
        self.metrics_df = self.metrics_df.sort_values(by=sort_columns)
        
        # Create internal group weights
        if group_cols:
            self.groups_df = self.metrics_df[self.group_cols].drop_duplicates().reset_index(drop=True)
            self.groups_df.insert(0, 'Group ID', ['G' + str(idx) for idx in range(1, len(self.groups_df) + 1)])

            # Create internal group weights
            for group_col in group_cols:
                temp_df = pd.DataFrame(metrics_df[group_col].drop_duplicates())
                temp_df['Weight'] = np.nan  # Initialize weight to NaN for each member of the group

                # Populate the group weights if given
                for idx, member in temp_df.iterrows():
                    member_name = member[group_col]
                    if member_name != 'ALL' and group_weights and group_col in group_weights \
                            and isinstance(group_weights[group_col], dict) \
                            and member_name in group_weights[group_col]:
                        temp_df.at[idx, 'Weight'] = group_weights[group_col][member_name]

                # Exclude 'ALL' members from count
                temp_df = temp_df[temp_df[group_col] != 'ALL']
                
                # Calculate the sum of defined weights and count of remaining NaN values
                sum_defined_weights = temp_df['Weight'].sum()
                remaining_members = temp_df[temp_df['Weight'].isna()]
                remaining_count = len(remaining_members)
                
                # Distribute the remaining weight equally among the members whose weights are not defined
                if remaining_count > 0:
                    remainder = 1 - sum_defined_weights
                    equal_weight = remainder / remaining_count
                    temp_df.loc[temp_df['Weight'].isna(), 'Weight'] = equal_weight
                    print(f"Remaining weights distributed equally among members of group column '{group_col}'.")

                self.group_weights[group_col] = temp_df

        
        # Get unique set of samples if unc_cols is provided and reset index
        if self.unc_cols:
            self.unc_smpls_df = self.metrics_df[self.unc_cols].drop_duplicates().reset_index(drop=True)
            self.unc_smpls_df.insert(0, 'Sample ID', ['S' + str(idx) for idx in range(1, len(self.unc_smpls_df)+1)])
        
        # Calculate criteria weights if not provided
        # Assumes equal weights if not provided 
        if not self.weights:
            # Allow for max to decimals for weights
            def custom_round(value):
                decimal_count = len(str(value).split(".")[1]) if "." in str(value) else 0
                decimals = 2 if decimal_count >= 2 else 1
                return round(value, decimals)  
            self.weights = {crit: custom_round(1/len(self.crit_cols)) for crit in self.crit_cols}
        # Check for duplicate rows
        if self.metrics_df.duplicated(subset=self.alt_cols + self.group_cols + self.unc_cols, keep=False).any():
            raise ValueError("Duplicated rows of alt_cols, group_cols, and sample. Some alternative IDs are counted more than once for some group ID and sample ID pairs.")

        
        # Create crit_df (DataFrame containing criteria data)
        data = []
        if self.crit_cols:
            for idx, criteria in enumerate(self.crit_cols):
                data.append({'Criteria ID': 'C' + str(idx + 1), 'Criteria': criteria, 'Weight': self.weights[criteria], 'Objective': self.objectives[criteria]})
        self.crit_df = pd.DataFrame(data)
        
        # Create cat_crit_df (DataFrame containing categorized criteria data)
        if not self.crit_cats:
            # If no categories are provided, create a default category for each criterion
            self.crit_cats = {crit: [crit] for crit in self.crit_cols}
        else:
            # Ensure all criteria are accounted for: if not included in any category, assign them to their own category
            uncategorized_criteria = set(self.crit_cols) - set(
                crit for crit_list in self.crit_cats.values() for crit in crit_list
            )
            for crit in uncategorized_criteria:
                self.crit_cats[crit] = [crit]

        # Create a DataFrame for categorized criteria
        data = []
        for idx, (category, criteria_list) in enumerate(self.crit_cats.items()):
            for criteria in criteria_list:
                data.append({'Cat ID': f'CAT{idx + 1}', 'Category': category, 'Criteria': criteria})

        self.cat_crit_df = pd.DataFrame(data)

        # Merge with crit_df to include additional criteria attributes (if any)
        self.cat_crit_df = self.cat_crit_df.merge(self.crit_df, on='Criteria', how='left')
        
        # Create alternatives_df (DataFrame containing alternatives data)
        if self.alt_cols:
            self.alternatives_df = self.metrics_df[self.alt_cols].drop_duplicates()
            self.alternatives_df.insert(0, 'Alternative ID', ['A' + str(idx) for idx in range(1, len(self.alternatives_df)+1)])
        else:
            raise ValueError("No alternative column given.")
        
        # Merge alternatives_df with groups and sample to create dm_df (Initialized DataFrame for decision-making)
        self.dm_df = self.alternatives_df.copy()
        self.dm_df['_merge'] = 1
        
        if isinstance(self.groups_df, pd.DataFrame):
            self.groups_df['_merge'] = 1
            self.dm_df = self.dm_df.merge(self.groups_df, on='_merge')
            self.groups_df = self.groups_df.drop('_merge', axis=1)
        
        if isinstance(self.unc_smpls_df, pd.DataFrame):
            self.unc_smpls_df['_merge'] = 1
            self.dm_df = self.dm_df.merge(self.unc_smpls_df, on='_merge')
            self.unc_smpls_df = self.unc_smpls_df.drop('_merge', axis=1)
        
        self.dm_df = self.dm_df.drop('_merge', axis=1)
        self.dm_df = pd.merge(self.dm_df, self.metrics_df[self.alt_cols + self.group_cols + self.unc_cols + self.crit_cols],
                             on=self.alt_cols + self.group_cols + self.unc_cols, how='left')
        

       # Generate default colors
        colormap = plt.cm.get_cmap('tab10', max(len(self.crit_cols), len(self.alternatives_df)))
        default_colors = [np.array(colormap(i)[:3]) for i in range(max(len(self.crit_cols), len(self.alternatives_df)))]

        # Generate colors for criteria (mapped to the criteria names from crit_cols)
        criteria_colors = {
            crit: default_colors[i % len(default_colors)] 
            for i, crit in enumerate(self.crit_cols)
        }

        # Generate or use input colors for alternatives (mapped to Alternative ID)
        alt_colors = {}
        if alternative_colors:
            for idx, row in self.alternatives_df.iterrows():
                alt_id = row['Alternative ID']
                # Use the provided color if available, else assign a default color
                alt_colors[alt_id] = alternative_colors.get(alt_id, default_colors[idx % len(default_colors)])
        else:
            # Generate default colors for all alternatives
            alt_colors = {
                row['Alternative ID']: default_colors[idx % len(default_colors)] 
                for idx, row in self.alternatives_df.iterrows()
            }

        # Store colors in a two-level dictionary
        self.colors = {
            "criteria": criteria_colors,
            "alternatives": alt_colors
        }

#%% Plottting methods
    # Method for plotting the criteria
    def plot_criteria_weights(self, group_by_category=True):
        """
        Plots the weights of criteria.

        Parameters:
        - group_by_category (bool): If True, the criteria will be grouped by category and displayed as a stacked bar plot. 
                    If False, the criteria will be displayed as individual bars.

        Returns:
        None
        """
        if group_by_category:
            # Create a bar plot from the pivoted DataFrame
            df = self.cat_crit_df.pivot(index='Category', columns='Criteria', values='Weight')
            ax = df.plot(
                kind='bar', 
                stacked=True, 
                figsize=(15, 8), 
                color=[self.colors['criteria'][crit] for crit in df.columns]  # Use mapped colors for criteria
            )
            # Calculate total weights for each category
            category_totals = df.sum(axis=1)

            # Annotate the total weights on top of each category
            for i, total in enumerate(category_totals):
                ax.text(
                    i, total,  # Slightly above the bar
                    f'Total: {round(total, 2)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold'
                )

            # Create an array to store the cumulative height of the bars
            cumulative_height = np.zeros(len(df))
            # Iterate over each bar (patch) in the plot
            for i, p in enumerate(ax.patches):
                # Calculate the index of the current bar in its stack
                bar_index = i % len(df)
                # Update the cumulative height of the bars in the current stack
                cumulative_height[bar_index] += p.get_height()
                # Only annotate bars with a height greater than zero
                if p.get_height() > 0:
                    # Annotate the height (weights value) of each bar on the plot
                    ax.annotate(
                        str(round(p.get_height(), 2)), 
                        (p.get_x() + p.get_width() / 2., cumulative_height[bar_index] - p.get_height() / 2), 
                        ha='center', va='center'
                    )

            # Set the x-axis label
            ax.set_xlabel('Criteria Categories', fontsize=12)
            # Set the x-axis labels to be truncated and tilted
            ax.set_xticklabels([label[:10] for label in df.index], rotation=45)
            # Set the legend
            plt.legend(
                bbox_to_anchor=(0., 1.02, 1., .102), 
                loc='lower left', 
                ncol=4, 
                mode="expand", 
                borderaxespad=0., 
                edgecolor='black', 
                title='Criteria', 
                fontsize=12
            )

        else:
            ax = self.crit_df.plot(
                x='Criteria', 
                y='Weight', 
                kind='bar', 
                figsize=(15, 8), 
                color=[self.colors['criteria'][crit] for crit in self.crit_df['Criteria']],  # Use mapped colors for criteria
                title='Weight of Criteria', 
                legend=False
            )
            for p in ax.patches:
                # Only annotate bars with a height greater than zero
                if p.get_height() > 0:
                    # Annotate the height (weights value) of each bar on the plot
                    ax.annotate(
                        str(round(p.get_height(), 2)), 
                        (p.get_x() + p.get_width() / 2., p.get_height() / 2), 
                        ha='center', va='center'
                    )
            ax.set_xlabel('Criteria', fontsize=12)
            # Set the x-axis labels to be truncated and tilted
            ax.set_xticklabels([label[:10] for label in self.crit_df['Criteria']], rotation=45)

        ax.set_ylabel('Weight', fontsize=12)
        ax.set_axisbelow(True)
        ax.grid(True, linestyle=':')
        plt.tight_layout()
        plt.show()


    

#%% Ranking methods
    # Method for ranking the alternatives
    # def calc_rankings(
    #         self,
    #         mcdm_methods=MCDM_DEFAULT,
    #         comp_ranks=COMP_DEFAULT,
    #         constraints={},
    #         derived_columns=None,
    #         use_dominance_as_constraint=False
    #     ):
    #     """
    #     Calculate rankings for a DecisionMatrix instance using specified Multi-Criteria Decision Making (MCDM) methods.

    #     Parameters:
    #     - mcdm_methods : Dict[str, MCDMMethod]
    #         Dictionary of MCDM methods to use for ranking the alternatives. 
    #     - comp_ranks : Dict[str, CompromiseRanking]
    #         Dictionary of compromise ranking methods to use for ranking the alternatives. 
    #     - constraints : Dict[str, str]
    #         Dictionary of constraints to apply to the decision matrix. 
    #     - derived_columns : List[str]
    #         List of derived columns to include in the output DataFrame.
    #     - use_dominance_as_constraint : bool
    #         If True, use dominance analysis to exclude dominated alternatives from the ranking process.

    #     Returns:
    #     - RanksOutput
    #         Object containing DataFrames with rankings, criteria rankings, and excluded alternatives.
    #     """
    #     # Prepare criteria weights and objectives
    #     weights = np.array([self.weights[crit_col] for crit_col in self.crit_cols])
    #     types = np.array([self.objectives[crit_col] for crit_col in self.crit_cols])
    #     red_dm_df = self.dm_df.copy()

    #     # Add 'Group ID' and 'Sample ID' if missing
    #     if 'Group ID' not in red_dm_df.columns:
    #         red_dm_df['Group ID'] = 'G1'
    #     if 'Sample ID' not in red_dm_df.columns:
    #         red_dm_df['Sample ID'] = 'S1'

    #     # Containers for results
    #     base_cols = list(self.alternatives_df.columns)
    #     if isinstance(self.groups_df, pd.DataFrame):
    #         base_cols += list(self.groups_df.columns)
    #     if isinstance(self.unc_smpls_df, pd.DataFrame):
    #         base_cols += list(self.unc_smpls_df.columns)

    #     alt_exc_nan_df = pd.DataFrame(columns=self.dm_df.columns)
    #     alt_exc_const_df = pd.DataFrame(columns=base_cols + list(constraints.keys()) + ['Dominance Constraint'])
    #     ranks_crit_df = pd.DataFrame(columns=base_cols + self.crit_cols)
    #     ranks_MCDM_df = pd.DataFrame(columns=base_cols + list(mcdm_methods.keys()) + list(comp_ranks.keys()))

    #     # Analyze dominance if required
    #     if use_dominance_as_constraint:
    #         dominance_pareto_df = self.analyze_dominance_and_pareto(constraints=constraints, derived_columns=derived_columns)

    #     # Iterate through each 'Group ID' and 'Sample ID' combination
    #     for _, group_scen_df in red_dm_df[['Group ID', 'Sample ID']].drop_duplicates().iterrows():
    #         sg_df = red_dm_df[
    #             (red_dm_df['Group ID'] == group_scen_df['Group ID']) &
    #             (red_dm_df['Sample ID'] == group_scen_df['Sample ID'])
    #         ]

    #         # Apply dominance constraint
    #         if use_dominance_as_constraint:
    #             group_sample_dominance_df = dominance_pareto_df[
    #                 (dominance_pareto_df['Group ID'] == group_scen_df['Group ID']) &
    #                 (dominance_pareto_df['Sample ID'] == group_scen_df['Sample ID'])
    #             ]
    #             dominated_alternatives = group_sample_dominance_df[group_sample_dominance_df['Pareto Optimal'] == False][self.alt_cols[0]].tolist()
    #             sg_df['Dominance Constraint'] = ~sg_df[self.alt_cols[0]].isin(dominated_alternatives)
    #         else:
    #             sg_df['Dominance Constraint'] = True

    #         # Handle NaN values
    #         nan_alt_rows = sg_df[self.crit_cols].isna().any(axis=1)
    #         if nan_alt_rows.any():
    #             alt_exc_nan_df = pd.concat([alt_exc_nan_df, sg_df[nan_alt_rows]], ignore_index=True)
    #             sg_df = sg_df[~nan_alt_rows]

    #         # Apply other constraints
    #         if constraints:
    #             sg_df, boolean_df = filter_dataframe(sg_df, filter_conditions=constraints, derived_columns=derived_columns, base_cols=base_cols)
    #             boolean_df['Dominance Constraint'] = sg_df['Dominance Constraint']
    #             alt_exc_const_df = pd.concat([alt_exc_const_df, boolean_df[~(boolean_df[list(constraints.keys()) + ['Dominance Constraint']] == True).all(axis=1)]], ignore_index=True)

    #         # Prepare matrix for MCDM methods
    #         matrix_df = sg_df[self.crit_cols].copy()
    #         matrix_df = matrix_df.apply(lambda col: col + abs(col.min()) + 1)
    #         matrix = matrix_df.to_numpy() + np.random.rand(*matrix_df.shape) * 1e-4

    #         # Calculate rankings if valid
    #         if matrix.any():
    #             temp_ranks_MCDM_df = sg_df[base_cols].copy()
    #             for method_name, method_instance in mcdm_methods.items():
    #                 if isinstance(method_instance, VIKOR) and matrix.shape[0] < 3:
    #                     print(f"Warning: VIKOR requires at least 3 alternatives. Skipping for {group_scen_df}.")
    #                     continue
    #                 elif not isinstance(method_instance, SPOTIS):
    #                     pref = method_instance(matrix, weights, types)
    #                 else:
    #                     bounds = np.vstack((np.amin(matrix, axis=0), np.amax(matrix, axis=0)))
    #                     pref = method_instance(matrix, weights, types, bounds)

    #                 reverse = not isinstance(method_instance, (VIKOR, SPOTIS))
    #                 temp_ranks_MCDM_df[method_name] = rank_preferences(pref, reverse=reverse)

    #             for comp_rank_name, comp_rank_instance in comp_ranks.items():
    #                 temp_ranks_MCDM_df[comp_rank_name] = comp_rank_instance(temp_ranks_MCDM_df[mcdm_methods.keys()].to_numpy())

    #             if ranks_MCDM_df.empty:
    #                 ranks_MCDM_df = temp_ranks_MCDM_df
    #             else:
    #                 ranks_MCDM_df = pd.concat([ranks_MCDM_df, temp_ranks_MCDM_df], ignore_index=True)

    #             ranks_crit_df = pd.concat([ranks_crit_df, ranks_columns(sg_df, columns=self.crit_cols, objectives=self.objectives)], ignore_index=True)

    #     if ranks_MCDM_df.empty:
    #         print("No alternatives to rank.")
    #         return None

    #     # Ensure Group ID and Sample ID are present
    #     if 'Group ID' not in base_cols:
    #         ranks_MCDM_df['Group ID'] = 'G1'
    #     if 'Sample ID' not in base_cols:
    #         ranks_MCDM_df['Sample ID'] = 'S1'

    #     return RanksOutput(ranks_crit_df, ranks_crit_df, ranks_MCDM_df, alt_exc_nan_df, alt_exc_const_df, list(mcdm_methods.keys()), list(comp_ranks.keys()), self)

            

    def calc_rankings(self, mcdm_methods = MCDM_DEFAULT, comp_ranks=COMP_DEFAULT, constraints ={}, rank_filt = {}, derived_columns = None):
            """
            Calculate rankings for a DecisionMatrix instance using specified Multi-Criteria Decision Making (MCDM) methods.
        
            Parameters:
            - mcdm_methods : Dict[str, MCDMMethod]
                Dictionary of MCDM methods to use for ranking the alternatives. 
                The keys are the names of the methods and the values are instances of the MCDMMethod class.
            - comp_ranks : Dict[str, CompromiseRanking]
                Dictionary of compromise ranking methods to use for ranking the alternatives. 
                The keys are the names of the methods and the values are instances of the CompromiseRanking class.
            - constraints : Dict[str, str]
                Dictionary of constraints to apply to the decision matrix. 
                The keys are the column names and the values are the conditions to apply.
            - rank_filt : Dict[str, str]
                Dictionary of filters to apply to the decision matrix before ranking. 
                The keys are the column names and the values are the conditions to apply.
            - derived_columns : List[str]
                List of derived columns to include in the output DataFrame.

            Returns:
            - ranks_df : pd.DataFrame
                DataFrame containing the rankings of the alternatives.
                
            """
            
            # provide criteria weights in array numpy.darray. All weights must sum to 1.
            weights = np.array([self.weights[crit_col] for crit_col in self.crit_cols])
            # provide criteria types in array numpy.darray. Profit criteria are represented by 1 and cost criteria by -1.
            types = np.array([self.objectives[crit_col] for crit_col in self.crit_cols])

            # Create a copy of the decision matrix DataFrame
            red_dm_df = self.dm_df.copy()
            # Check if both 'Group ID' and 'Sample ID' columns exist
            if 'Group ID' not in red_dm_df.columns:
                red_dm_df['Group ID'] = 'G1'
            if 'Sample ID' not in red_dm_df.columns:
                red_dm_df['Sample ID'] = 'S1'
            # Pre-filter which sceanrio, groups and 
            red_dm_df, _ = filter_dataframe(red_dm_df, filter_conditions=rank_filt, derived_columns=derived_columns)
            
            ## Create data frames to store data
            # Define base column
            base_cols = list(self.alternatives_df.columns)
            if isinstance(self.groups_df, pd.DataFrame):
                base_cols +=  list(self.groups_df.columns)
            if isinstance(self.unc_smpls_df, pd.DataFrame):
                base_cols += list(self.unc_smpls_df.columns)
            # Alternatives not included
            alt_exc_nan_df = pd.DataFrame(columns= self.dm_df.columns) # To store nan alternatives
            alt_exc_const_df = pd.DataFrame(columns= base_cols+ list(constraints.keys())) # To store nan alternatives
            # rank containers
            ranks_crit_df = pd.DataFrame(columns=base_cols + self.crit_cols)
            #ranks_mcdm_methods_df = pd.DataFrame(columns=base_cols + list(mcdm_methods.keys()))
            #ranks_comp_df = pd.DataFrame(columns=base_cols + list(comp_ranks.keys()))
            ranks_MCDM_df = pd.DataFrame(columns=base_cols + list(mcdm_methods.keys()) + list(comp_ranks.keys()))

            # Check if crit_cols contains any zero values
            if red_dm_df[self.crit_cols].isin([0]).any().any():
                # Iterate through MCDM methods
                for method_name, method_instance in mcdm_methods.items():
                    if isinstance(method_instance, (ARAS, CODAS, CRADIS)):
                        print(f"Warning: {method_name} is of type {type(method_instance)}, which may require special handling due to zero values in some criteria columns. Recmonedeation is to replace the zero values with negligaibel numbers.", 3*"...\n")
            

            # Iterate through all pairs of 'Group ID' and 'Sample ID'
            for _, group_scen_df in red_dm_df[['Group ID', 'Sample ID']].drop_duplicates().iterrows():
            
                # Check if both columns exist
                sg_df = red_dm_df[red_dm_df[['Group ID', 'Sample ID']].isin(group_scen_df[['Group ID', 'Sample ID']].values).all(axis=1)]

                # Store all not included alternatives
                # due to NaN values
                nan_alt_rows = sg_df[self.crit_cols].isna().any(axis=1)
                if nan_alt_rows.any():
                    # Check if empty or all-NA rows
                    if alt_exc_nan_df.empty:
                        alt_exc_nan_df = sg_df[nan_alt_rows]
                    else:
                        alt_exc_nan_df = pd.concat([alt_exc_nan_df, sg_df[nan_alt_rows]], ignore_index=True)
                    sg_df.reset_index(drop=True, inplace=True)
                    nan_alt_rows.reset_index(drop=True, inplace=True)
                    sg_df = sg_df[~nan_alt_rows]
                    
                # 
                if constraints:
                    sg_df, boolean_df = filter_dataframe(sg_df, filter_conditions=constraints, derived_columns=derived_columns, base_cols=base_cols)
                    alt_exc_const_df = pd.concat([alt_exc_const_df, boolean_df[~(boolean_df[constraints.keys()]==True).all(axis=1)]], ignore_index=True)


                # Find the smallest number in each column and add its absolute value plus one to the column
                matrix_df = sg_df[self.crit_cols].copy()
                matrix_df = matrix_df.apply(lambda col: col + abs(col.min()) + 1)

                # Convert the DataFrame back to a numpy array
                matrix = matrix_df.to_numpy()

                # Add a random small positive value to each element of the matrix
                matrix = matrix + np.random.rand(*matrix.shape) * 1e-4
                
                # Flag to track whether VIKOR warning has been printed
                vikor_warning_flag = False

                # Check if matrix is not empty and contains any non-zero values (i.e., not all-zero)
                if matrix.any():
                    
                    # Temp container
                    temp_ranks_MCDM_df = sg_df[base_cols].copy()
            
                    ## Calc ranking for each MCDM method
                    for pipe in mcdm_methods.keys():
            
                        # Calculate the preference values of alternatives
                        if isinstance(mcdm_methods[pipe],VIKOR) and matrix.shape[0] < 3: # VIKOR requires at least 3 alternatives
                            # set pref as nan array
                            if not vikor_warning_flag:
                                print(  f"Warning: VIKOR requires at least 3 alternatives. "
                                        f"In this dataset, some groups or samples have fewer alternatives, and VIKOR rankings for those cases will be skipped."
                                    )
                                vikor_warning_flag = True  # Set the flag to True after printing the warning
                            pref = np.full(matrix.shape[0], np.nan)
                        elif not isinstance(mcdm_methods[pipe],SPOTIS):
                            pref = mcdm_methods[pipe](matrix, weights, types)
                        else:
                            # SPOTIS preferences must be sorted in ascending order
                            bounds_min = np.amin(matrix, axis = 0)
                            bounds_max = np.amax(matrix, axis = 0)
                            bounds = np.vstack((bounds_min, bounds_max))
                            # Calculate the preference values of alternatives
                            pref = mcdm_methods[pipe](matrix, weights, types, bounds)
                            
                        # Generate ranking of alternatives by sorting alternatives descendingly according to the TOPSIS algorithm (reverse = True means sorting in descending order) according to preference values
                        if  isinstance(mcdm_methods[pipe], (MULTIMOORA)):
                            temp_ranks_MCDM_df.loc[~nan_alt_rows, pipe] = mcdm_methods[pipe](matrix, weights, types) # Mu;timoora includes ranker
                        elif isinstance(mcdm_methods[pipe], (VIKOR, SPOTIS)):
                            temp_ranks_MCDM_df.loc[~nan_alt_rows, pipe] = rank_preferences(pref, reverse = False)
                        else:
                            temp_ranks_MCDM_df.loc[~nan_alt_rows, pipe] = rank_preferences(pref, reverse = True)
                    
                    # Calc compromised ranking
                    if comp_ranks:
                        for comp_rank in comp_ranks.keys():
                            temp_ranks_MCDM_df.loc[~nan_alt_rows, comp_rank] = comp_ranks[comp_rank](temp_ranks_MCDM_df.loc[~nan_alt_rows,mcdm_methods.keys()].to_numpy())
                    
        
                    # Populate the containers
                    # Exclude empty or all-NA columns before concatenation
                    # Check if temp_ranks_MCDM_df is empty
                    temp_ranks_MCDM_df = temp_ranks_MCDM_df.dropna(how='all', axis=1)
                    if temp_ranks_MCDM_df.empty:
                        pass
                    elif ranks_MCDM_df.empty:
                        ranks_MCDM_df = temp_ranks_MCDM_df
                    else:
                        ranks_MCDM_df = pd.concat([ranks_MCDM_df, temp_ranks_MCDM_df], ignore_index=True)
                    ranks_crit_df = pd.concat([ranks_crit_df, ranks_columns(sg_df, columns=self.crit_cols, objectives=self.objectives)], ignore_index=True) # Calc criteria ranking
                    ranks_df = pd.merge(ranks_crit_df, ranks_MCDM_df)
        
            # Check if ranks_MCDM_df is empty
            if ranks_MCDM_df.empty:
                print("No alternatives to rank.")
                return
            # Store all not included alternatives with ranking zero
            # Get all columns from list(mcdm_methods.keys()) that have a value of zero in ranks_MCDM_df
            #zero = ranks_MCDM_df.columns[ranks_MCDM_df.isin([0]).any()].tolist()
                
            
            # TODO: Quick fix to add Group ID and Sample ID to the ranks_MCDM_df
            if 'Group ID' not in base_cols:
                ranks_MCDM_df['Group ID'] = 'G1'
            if 'Sample ID' not in base_cols:
                ranks_MCDM_df['Sample ID'] = 'S1'

            return RanksOutput(ranks_df, ranks_crit_df, ranks_MCDM_df, alt_exc_nan_df, alt_exc_const_df, list(mcdm_methods.keys()), list(comp_ranks.keys()), self)

#%% Methods fro manipulating the DecisionMatrix object (e.g., pivoting, reweighting, etc.)

    # Method to pivot and reweight the critera based on the group column and group weights
    def pivot_and_reweight_criteria(self, piv_col):
            """
           Pivot and reweight criteria based on a specified pivot column and group weights.
    
           Parameters:
           - piv_col: str
               The column name to pivot the criteria data.
    
           Returns:
           - new_self: DecisionMatrix
               A new instance of DecisionMatrix with pivoted criteria.
           """
            

            # Define pivot and index columns for pivot
            index_col = [col for col in self.alt_cols + self.unc_cols + self.group_cols + self.crit_cols if col not in self.crit_cols + [piv_col]]

            # Filter out rows where the specified column is ALL or nan
            filt_dm_df = self.dm_df[self.alt_cols + self.unc_cols + self.group_cols + self.crit_cols]
            filt_dm_df = filt_dm_df[~filt_dm_df[piv_col].isin(['ALL']) & filt_dm_df[piv_col].notna()]


            crit_piv_df = filt_dm_df.pivot(index = index_col,
                                          columns = piv_col,
                                          values = self.crit_cols)

            crit_piv_df = crit_piv_df.reset_index()
            crit_piv_df.columns = [f'{"_".join(col)}' if col[1] else f'{col[0]}' for col in crit_piv_df.columns]


            # Step 3: Remove duplicates and create a copy of weights
            new_weights = copy.deepcopy(self.weights)
            new_crit_cats = {key: [] for key in self.crit_cats.keys()}

            group_values = list(filt_dm_df[piv_col].dropna().drop_duplicates())
            new_objectives = copy.deepcopy(self.objectives) # Initialize new objectives for pivoted criteria

            for crit_col in self.crit_cols:
                new_crit_cols_temp = [crit_col + '_' + group_value for group_value in group_values]
                temp_df = crit_piv_df[new_crit_cols_temp]

                cat_crit = self.cat_crit_df[self.cat_crit_df['Criteria'].isin([crit_col])]['Category'].values[0]
                
                # Step 4: Check if all columns have the same values
                if temp_df.apply(lambda col: col.equals(temp_df.iloc[:, 0])).all():
                    print(f"{crit_col}: All columns have the same values. Retain the original name.")
                    crit_piv_df = crit_piv_df.rename(columns={new_crit_cols_temp[0]: crit_col})
                    if len(new_crit_cols_temp) > 1:
                        crit_piv_df = crit_piv_df.drop(columns=new_crit_cols_temp[1:])
                    # Update cat crits
                    new_crit_cats[cat_crit].append(crit_col)
                else:
                    print(f"{crit_col}: Columns have different values. Reweight and introduce new criteria.")
                    for group_value in group_values:
                        idx = self.group_weights[piv_col][piv_col].isin([group_value])
                        group_weight = self.group_weights[piv_col]['Weight'][idx].values[0]
                        new_weights.update({crit_col + '_' + group_value: new_weights[crit_col]*group_weight})
                        new_objectives[crit_col + '_' + group_value] = self.objectives[crit_col]

                        # Update cat crits
                        new_crit_cats[cat_crit].append(crit_col + '_' + group_value)
                        
                    del new_weights[crit_col]
                    del new_objectives[crit_col]

            # Step 4 (continued): Reorder the dictionary as per crit_piv_df columns
            new_weights = {key: new_weights[key] for key in crit_piv_df.columns if key in new_weights.keys()}

            # Step 4 (continued): Reorder the dictionary as per crit_piv_df columns
            new_objectives = {key: new_objectives[key] for key in crit_piv_df.columns if key in new_objectives.keys()}

            # Step 4 (continued): Reorder the dictionary as per crit_piv_df columns
            new_group_cols = [col for col in self.group_cols if col != piv_col]
            new_crit_cols = list(new_objectives.keys())

            # Step 5: Remove the group key from the group weights
            new_group_weights = {key: value for key, value in self.group_weights.items() if key != piv_col}
   
    
            # Create a new DecisionMatrix instance with modified attributes
            new_self = DecisionMatrix(
                metrics_df=crit_piv_df,
                objectives=new_objectives,
                alt_cols=self.alt_cols,
                crit_cols=new_crit_cols,
                weights=new_weights,
                group_cols=new_group_cols,
                unc_cols=self.unc_cols,
                #unc_var_prob_dist=self.unc_var_prob_dist,
                crit_cats=new_crit_cats,
                group_weights=new_group_weights,
                # Include other necessary attributes for initialization of the new instance
            )
    
            return new_self

    # Method to calculate the conditional criteria
    def mean_based_criteria(self, condition={}, derived_columns=None):
        '''
        Calculate the mean of the criteria based on the condition provided.

        Parameters:
        - condition: dict
            Dictionary of conditions to apply to the decision matrix. 
            The keys are the column names and the values are the conditions to apply.
        - derived_columns: list
            List of derived columns to include in the output DataFrame.

        Returns:
        - new_self: DecisionMatrix
            A new instance of DecisionMatrix with mean-based criteria.
        '''
        dm_df = self.dm_df.copy()

        # Define base column
        base_cols = list(self.alternatives_df.columns) + ['Group ID']
        if isinstance(self.groups_df, pd.DataFrame):
            base_cols += list(self.groups_df.columns)
        else:
            dm_df['Group ID'] = 'G1'

        # Remove duplicates in base_cols
        base_cols = list(dict.fromkeys(base_cols))

        # Create a dataframe to store the results
        new_dm_df = pd.DataFrame(columns=base_cols + self.crit_cols)

        # Apply constraints per group and state combo
        for _, alt_group_df in dm_df[['Alternative ID', 'Group ID']].drop_duplicates().iterrows():
            # Filter the group and state
            sg_df = dm_df[dm_df[['Alternative ID', 'Group ID']].isin(alt_group_df[['Alternative ID', 'Group ID']].values).all(axis=1)]
            
            # Filter the dataframe based on the condition 
            filt_sg_df,_ = filter_dataframe(sg_df, filter_conditions=condition, derived_columns=derived_columns)

            # Print the alternative and group that are all filtered out
            if filt_sg_df.empty:
                print(f"The alternative {alt_group_df['Alternative ID']} in group {alt_group_df['Group ID']} did not satisfy the condition and is filtered out.")
                continue

            # Calculate the mean of the criteria
            mean_crits_temp_df = filt_sg_df[self.crit_cols].mean()

            # Add the mean_crits_temp_df columns and results to the base dataframe
            base_temp_df = sg_df[base_cols].drop_duplicates()
            base_temp_df = base_temp_df.assign(**mean_crits_temp_df)

            # add the base_temp_df to the new_dm_df
            if new_dm_df.empty:
                new_dm_df = base_temp_df
            else:
                new_dm_df = pd.concat([new_dm_df, base_temp_df], ignore_index=True)

        # If only one group is present, remove the group column
        if len(new_dm_df['Group ID'].unique()) == 1:
            new_dm_df = new_dm_df.drop(columns=['Group ID'])

        # Create a new DecisionMatrix instance with modified attributes
        new_self = DecisionMatrix(
            metrics_df = new_dm_df,
            objectives=self.objectives,
            alt_cols= self.alt_cols,
            crit_cols= self.crit_cols,
            weights = self.weights,
            group_cols = self.group_cols,
            crit_cats = self.crit_cats,
            group_weights = self.group_weights,
        )
        
        return new_self
    
    def _plot_combined_sensitivity(self, merged_ranks_df, merged_imp_sens_df):
        """
        Plot combined sensitivity analysis results for multiple targets.

        Parameters:
            merged_ranks_df (DataFrame): Rankings DataFrame for all targets.
            merged_imp_sens_df (DataFrame): Sensitivity weights DataFrame for all targets.
        """
        unique_targets = merged_imp_sens_df['Target'].unique()
        num_targets = len(unique_targets)

        # Create subplots for weights and ranks
        fig, axes = plt.subplots(2, num_targets, figsize=(6 * num_targets, 10), squeeze=False)

        for i, target in enumerate(unique_targets):
            # Filter data for the current target
            weights_data = merged_imp_sens_df[merged_imp_sens_df['Target'] == target].drop(columns=['Target'])
            ranks_data = merged_ranks_df[merged_ranks_df['Target'] == target]

            # Reset index to ensure the 'index' column exists
            weights_data = weights_data.reset_index()

            # Plot sensitivity weights (bar plot)
            weights_data.set_index('index').plot(
                kind='bar',
                stacked=True,
                ax=axes[0, i],
                title=f"Sensitivity Weights for {target}",
                legend=(i == num_targets - 1),  # Show legend only for the last plot
            )
            axes[0, i].set_ylabel("Weight")
            axes[0, i].set_xlabel("Importance Level (%)")
            axes[0, i].tick_params(axis='x', rotation=45)

            # Plot rankings (line plot)
            pivoted_ranks = ranks_data.pivot(index='Weight', columns='Alternative', values='Rank')
            pivoted_ranks.plot(
                kind='line',
                ax=axes[1, i],
                title=f"Sensitivity Rankings for {target}",
                legend=(i == num_targets - 1),  # Show legend only for the last plot
            )
            axes[1, i].set_ylabel("Rank")
            axes[1, i].set_xlabel("Importance Level (%)")
            axes[1, i].invert_yaxis()  # Lower rank is better, so invert y-axis
            axes[1, i].grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.show()



#%% Sensitivity (weight) analysis 

    def analyse_sens_weights(self, targets=None, plot_ranks=True, plot_weight_dist=True, group_filter='G1', sample_filter='S1', **ranking_args):
        """
        Analyze sensitivity of weights for specified criteria, categories, or custom criteria sets.

        Parameters:
            targets (str, list, or dict, optional): Specifies the criteria or categories to analyze.
                - None: Analyze all predefined categories in `cat_crit_df`.
                - str: A single criterion or category name.
                - list: A list of criteria or categories (must not mix both).
                - dict: A mapping of custom categories, e.g., {'Custom Name': [list_of_criteria]}.
            group_filter (str): Group ID to filter on. Default is 'G1'.
            sample_filter (str): Sample ID to filter on. Default is 'S1'.
            plot_ranks (bool): If True, plots rank sensitivity graphs.
            plot_weight_dist (bool): If True, plots weight distribution graphs.
            **ranking_args: Additional arguments to pass to `_calc_imprt_sensitivity`.

        Returns:
            tuple: (merged_ranks_df, merged_imp_sens_df)
                - merged_ranks_df: DataFrame with merged rankings for all targets.
                - merged_imp_sens_df: DataFrame with merged sensitivity of criteria weights.
        """
        # Step 1: Filter decision matrix for group and sample
        filtered_dm_df = self.dm_df.copy()
        if 'Group ID' in filtered_dm_df.columns:
            filtered_dm_df = filtered_dm_df[filtered_dm_df['Group ID'] == group_filter]
        if 'Sample ID' in filtered_dm_df.columns:
            filtered_dm_df = filtered_dm_df[filtered_dm_df['Sample ID'] == sample_filter]

        if filtered_dm_df.empty:
            raise ValueError(f"No data available for Group ID '{group_filter}' and Sample ID '{sample_filter}'.")

        # Step 2: Validate and process `targets`
        if targets is None:
            # Analyze all predefined categories
            targets = {
                cat: self.cat_crit_df[self.cat_crit_df['Category'] == cat]['Criteria'].tolist()
                for cat in self.cat_crit_df['Category'].unique()
            }
        elif isinstance(targets, str):
            # Single string: Treat as either a category or a criterion
            if targets in self.cat_crit_df['Category'].unique():
                targets = {
                    targets: self.cat_crit_df[self.cat_crit_df['Category'] == targets]['Criteria'].tolist()
                }
            elif targets in self.crit_df['Criteria'].tolist():
                targets = {targets: [targets]}
            else:
                raise ValueError(f"Invalid target: '{targets}' is neither a valid category nor a criterion.")
        elif isinstance(targets, list):
            # List of either categories or criteria
            all_categories = self.cat_crit_df['Category'].unique()
            all_criteria = self.crit_df['Criteria'].tolist()

            if all(t in all_categories for t in targets):
                # All targets are categories
                targets = {
                    cat: self.cat_crit_df[self.cat_crit_df['Category'] == cat]['Criteria'].tolist()
                    for cat in targets
                }
            elif all(t in all_criteria for t in targets):
                # All targets are criteria
                targets = {crit: [crit] for crit in targets}
            else:
                # Mixed or invalid items
                invalid = [t for t in targets if t not in all_categories and t not in all_criteria]
                raise ValueError(f"The list must contain either all categories or all criteria (do not mix). Invalid items: {invalid}")
        elif isinstance(targets, dict):
            # Dictionary of custom categories
            for category, crits in targets.items():
                invalid_crits = [c for c in crits if c not in self.crit_df['Criteria'].tolist()]
                if invalid_crits:
                    raise ValueError(f"Invalid criteria in custom category '{category}': {invalid_crits}")
        else:
            raise ValueError("Invalid `targets` format. Must be None, str, list, or dict.")

        # Step 3: Analyze sensitivity for each target
        all_ranks = []
        all_weights = []

        for category, criteria in targets.items():
            print(f"Analyzing sensitivity for: {category} with criteria: {criteria}")

            # Call `_calc_imprt_sensitivity` for each target
            ranks_df, weights_df = self._calc_imprt_sensitivity(
                crit_target={category: criteria},
                plot_ranks=plot_ranks,
                plot_weight_dist=plot_weight_dist,
                **ranking_args
            )

            # Append results to containers
            ranks_df['Target'] = category
            weights_df['Target'] = category
            all_ranks.append(ranks_df)
            all_weights.append(weights_df)

        # Step 4: Merge results into DataFrames
        merged_ranks_df = pd.concat(all_ranks, ignore_index=True)
        merged_imp_sens_df = pd.concat(all_weights, ignore_index=True)

        return merged_ranks_df, merged_imp_sens_df





    def _calc_imprt_sensitivity(self, crit_target, plot_ranks=True, plot_weight_dist=True, **ranking_args):
        """
        Calculate and analyze sensitivity of importance weights for a single criterion or a group of criteria.

        Parameters:
            crit_target (str or dict): 
                - If str, specifies a single criterion whose weight is varied.
                - If dict, defines a group of criteria with the format {'Group Name': [list_of_criteria]}.
            plot_ranks (bool): If True, plots rank sensitivity graphs.
            plot_weight_dist (bool): If True, plots weight distribution graphs.
            **ranking_args: Additional arguments to pass to the `calc_rankings` method, including mcdm_methods and comp_ranks.

        Returns:
            tuple: (ranks_imp_df, imp_sens_df)
                - ranks_imp_df: DataFrame with rankings at different importance levels.
                - imp_sens_df: DataFrame with sensitivity of criteria weights.
        """

        # Dynamically determine the alternative column
        alt_name_col = self.alt_cols[0] if len(self.alt_cols) == 1 else "Alternative ID"

        # Get the criteria dataframe
        crit_df = self.crit_df

        # Hardcode weight distribution steps (0 to 1 in 11 steps)
        imp_tot = np.linspace(0, 1, 11)

        # Handle `crit_target` to determine the criteria to vary
        if isinstance(crit_target, str):  # Single criterion case
            target_criteria = [crit_target]
            xlabel = crit_target
        elif isinstance(crit_target, dict):  # Group case
            group_name, target_criteria = list(crit_target.items())[0]
            xlabel = group_name
        else:
            raise ValueError("crit_target must be either a string (single criterion) or a dictionary (group of criteria).")

        # Separate target and non-target criteria
        crit_group_df = crit_df[crit_df['Criteria'].isin(target_criteria)][['Criteria', 'Weight']]
        crit_non_group_df = crit_df[~crit_df['Criteria'].isin(target_criteria)][['Criteria', 'Weight']]

        # Calculate the current weight of the target group or criteria
        current_weight = crit_group_df['Weight'].sum()

        # Create a dataframe to store the sensitivity of the weights
        imp_sens_df = pd.DataFrame(index=imp_tot, columns=list(crit_group_df['Criteria'].unique()) + list(crit_non_group_df['Criteria'].unique()))

        # Iterate over the total weights values
        for imp in imp_tot:
            new_group_weights = crit_group_df.copy()
            new_group_weights['Weight'] = imp * crit_group_df['Weight'] / crit_group_df['Weight'].sum()
            new_non_group_weights = crit_non_group_df.copy()
            new_non_group_weights['Weight'] = (1 - imp) * crit_non_group_df['Weight'] / crit_non_group_df['Weight'].sum()
            imp_sens_df.loc[imp, new_group_weights['Criteria']] = new_group_weights['Weight'].values
            imp_sens_df.loc[imp, new_non_group_weights['Criteria']] = new_non_group_weights['Weight'].values

        # Extract ranking methods
        mcdm_methods = ranking_args.get('mcdm_methods', MCDM_DEFAULT)
        comp_ranks = ranking_args.get('comp_ranks', COMP_DEFAULT)

        # Determine the ranking method name
        if comp_ranks:
            rank_method_name = list(comp_ranks.keys())[0]
        elif len(mcdm_methods) == 1:
            rank_method_name = list(mcdm_methods.keys())[0]
        else:
            raise ValueError('You need to specify a compromise ranking method or only one MCDM method.')

        # Iterate over the rows in the weights sensitivity dataframe
        for idx, row in enumerate(imp_sens_df.iterrows()):
            new_weights = row[1].to_dict()
            dm_temp = DecisionMatrix(metrics_df=self.metrics_df, objectives=self.objectives, alt_cols=self.alt_cols, crit_cols=list(self.objectives.keys()), weights=new_weights)
            rank_obj_temp = dm_temp.calc_rankings(**ranking_args)

            if len(rank_obj_temp.ranks_df['Group ID'].unique()) != 1 or len(rank_obj_temp.ranks_df['Sample ID'].unique()) != 1:
                raise ValueError('The rank object contains more than one group or sample.')

            if idx == 0:
                ranks_imp_df = rank_obj_temp.ranks_df[[alt_name_col, rank_method_name]].copy()
                ranks_imp_df['Weight'] = row[0]
            else:
                ranks_imp_df_temp = rank_obj_temp.ranks_df[[alt_name_col, rank_method_name]].copy()
                ranks_imp_df_temp['Weight'] = row[0]
                ranks_imp_df = pd.concat([ranks_imp_df, ranks_imp_df_temp], axis=0)

        # Plot the sensitivity if required
        if plot_ranks:
            self._plot_rank_sens_weights(ranks_imp_df, alt_name_col, rank_method_name, xlabel, current_weight)
        if plot_weight_dist:
            self._plot_crit_weights_sensitivity(imp_sens_df, xlabel, current_weight)
        
        return ranks_imp_df, imp_sens_df


    def _plot_crit_weights_sensitivity(self, imp_sens_df, xlabel, current_weight):
        """
        Plots a stacked bar chart of criteria weights sensitivity.

        Parameters:
            imp_sens_df (DataFrame): Sensitivity data with weights for each criterion at different levels.
            xlabel (str): Label for the x-axis indicating the group or criterion being varied.
            current_weight (float): The current weight value of the group/criterion to be highlighted.
        """
        # Ensure colors are applied based on criteria
        criteria_colors = [self.colors['criteria'][crit] for crit in imp_sens_df.columns]

        # Plot the DataFrame as a stacked bar plot
        ax = imp_sens_df.plot(kind='bar', stacked=True, figsize=(12, 6), color=criteria_colors)

        # Set the x-axis label
        ax.set_xlabel(f'Weight of {xlabel}', fontsize=14)

        # Set the y-axis label
        ax.set_ylabel('Criteria Weight', fontsize=14)

        # Format x-axis ticks to display percentage
        ax.set_xticklabels([f'{int(tick * 100)}%' for tick in imp_sens_df.index], rotation=0, fontsize=12)

        # Format the y-axis labels as percentages
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

        # Highlight the current weight
        closest_idx = (abs(imp_sens_df.index - current_weight)).argmin()
        x_tick_label = f'{int(current_weight * 100)}%'
        ax.axvline(x=closest_idx, color='red', linestyle='--', linewidth=2, label=f'Current Weight: {x_tick_label}')
        ax.text(closest_idx, 1.05, x_tick_label, color='red', fontsize=12, ha='center')

        # Place the legend with proper title
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Criteria', fontsize=12)

        plt.tight_layout()
        plt.show()


    def _plot_rank_sens_weights(self, ranks_imp_df, alt_tag, rank_method_name, xlabel, current_weight):
        """
        Plot sensitivity of rankings with respect to varying weights.

        Parameters:
            ranks_imp_df (DataFrame): DataFrame with rankings at different weights.
            alt_tag (str): Column name for alternatives.
            rank_method_name (str): Column name for ranking method.
            xlabel (str): Label for x-axis.
            current_weight (float): The current weight value to highlight.
        """
        if not isinstance(ranks_imp_df, pd.DataFrame):
            raise ValueError("ranks_imp_df must be a pandas DataFrame")

        # Pivot the DataFrame to make each alternative a column
        plot_df = ranks_imp_df.pivot(index='Weight', columns=alt_tag, values=rank_method_name)

        # Create a mapping between descriptive names and 'A+' keys
        alt_mapping = {
            row[alt_tag]: row['Alternative ID']
            for _, row in self.alternatives_df.iterrows()
        }

        # Ensure colors align with the `A+` keys in `self.colors['alternatives']`
        color_dict = {}
        for col in plot_df.columns:
            alt_key = alt_mapping.get(col)  # Find the 'A+' key for the descriptive name
            if alt_key in self.colors['alternatives']:
                color_dict[col] = self.colors['alternatives'][alt_key]
            else:
                print(f"Warning: No color found for {col}. Assigning default color.")
                color_dict[col] = (0.5, 0.5, 0.5)  # Default gray color

        # Plot with proper colors
        ax = plot_df.plot(kind='line', grid=True, figsize=(12, 6), color=[color_dict[col] for col in plot_df.columns])

        # X and Y labels
        ax.set_xlabel(f'Total weights of {xlabel}', fontsize=14)
        ax.set_ylabel('Rank', fontsize=14)

        # Format x-axis labels
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

        # Limits
        ax.set_ylim(0, plot_df.max().max() + 1)
        ax.set_xlim(plot_df.index[0], plot_df.index[-1])

        # Highlight current weight
        closest_idx = (abs(plot_df.index - current_weight)).argmin()
        closest_value = plot_df.index[closest_idx]
        ax.axvline(x=closest_value, color='red', linestyle='--', linewidth=2, label=f'Current Weight: {int(current_weight * 100)}%')

        ax.text(closest_value, plot_df.max().max() + 1, f'{int(current_weight * 100)}%', color='red', fontsize=12, ha='center', va='bottom')

        # Generate legend data with ranks at [0% -> current -> 100%]
        legend_data = [
            {
                "label": f'{col} [{int(plot_df.iloc[0, i])}, {int(plot_df.iloc[closest_idx, i])}, {int(plot_df.iloc[-1, i])}]',
                "color": color_dict[col],
                "final_rank": int(plot_df.iloc[-1, i])  # Final rank at 100%
            }
            for i, col in enumerate(plot_df.columns)
        ]

        # Sort legend data by final rank (lowest at the bottom)
        legend_data = sorted(legend_data, key=lambda x: x["final_rank"], reverse=True)

        # Add a custom line for the "Current Weight" indicator
        current_weight_line = mlines.Line2D([], [], color='red', linestyle='--', label=f'Current Weight ({int(current_weight * 100)}%)')

        # Create legend handles with "Current Weight" first
        lines = [current_weight_line] + [
            mlines.Line2D([], [], color=item["color"], label=item["label"])
            for item in legend_data
        ]

        # Create legend
        legend_title = f'Rank at [0%, {int(current_weight * 100)}%, 100%]'
        plt.legend(
            handles=lines,
            bbox_to_anchor=(1.05, 0.5),
            loc='center left',
            borderaxespad=0.0,
            edgecolor='black',
            fontsize=12,
            title=legend_title,
        )

        plt.tight_layout()
        plt.show()











    #%% Print the DecisionMatrix object

    def print_dm(self):
        """Print the decision matrix and its attributes."""
        sections = {
            'Decision Matrix': self.dm_df,
            'Criteria DataFrame': self.crit_df,
            'Categorized Criteria DataFrame': self.cat_crit_df,
            'Alternatives DataFrame': self.alternatives_df,
            'Groups DataFrame': self.groups_df,
            'Uncertainty Samples': self.unc_smpls_df
        }
        
        # Helper function to format and print
        def safe_print(title, df):
            print(f'{title}:')
            if df is not None and not df.empty:
                print(tb.tabulate(df, headers='keys', tablefmt='pretty'))
            else:
                print("DataFrame is empty or not defined.")
            print('')

        # Print each section
        for title, df in sections.items():
            safe_print(title, df)
        
        # Print group weights separately
        print('Group Weights:')
        if self.group_weights:
            for group, weights in self.group_weights.items():
                print(f"\nGroup: {group}")
                # Check if weights are in DataFrame format
                if isinstance(weights, pd.DataFrame):
                    print(tb.tabulate(weights, headers='keys', tablefmt='pretty'))
                elif isinstance(weights, dict):
                    # Handle dictionary weights
                    print(tb.tabulate(weights.items(), headers=['Member', 'Weight'], tablefmt='pretty'))
                else:
                    print(f"Unsupported weight format for group '{group}': {weights}")
        else:
            print("Group weights are empty or not defined.")

#%% Plotting methods
    def plot_norm_criteria_values(self, norm_func=norms.linear_normalization, scale_to=None):
        """
        Normalize the decision matrix and plot the normalized criteria values with colors based on the 'colors' attribute.

        Parameters:
            norm_func (function): Normalization function to apply, default is linear_normalization.
            scale_to (int or None): If specified, rescales normalized values to a 1–scale_to range (e.g., 5 for a 1–5 scale).
        """
        # Copy the decision matrix
        modified_dm_df = self.dm_df.copy()

        # Set the alternative name column to the first column if the alt_cols is more than one
        alt_name_col = self.alt_cols[0] if self.alt_cols[0] != 'Alternative' else "Alternative ID"

        # Apply normalization to the decision matrix
        raw_dm = modified_dm_df[self.crit_cols].values
        objectives_array = np.array([self.objectives[col] for col in self.crit_cols])
        norm_values = norm_func(raw_dm, objectives_array)
        norm_df = pd.DataFrame(norm_values, columns=self.crit_cols)

        # Rescale to the specified range if `scale_to` is given
        if scale_to:
            norm_df = self.rescale_to_range(norm_df, scale_to)

        # Add the alternative column to the normalized DataFrame
        norm_df[alt_name_col] = modified_dm_df[alt_name_col]
        plot_df = norm_df

        # Reshape data for plotting
        plot_data = plot_df.melt(
            id_vars=[alt_name_col],
            value_vars=self.crit_cols,
            var_name='Criterion',
            value_name='Normalized Value'
        )

        # Map colors to criteria using self.colors['criteria']
        criteria_colors = [self.colors['criteria'][crit] for crit in plot_data['Criterion'].unique()]
        sns.set_palette(sns.color_palette(criteria_colors))

        # Plot the normalized criteria values
        plt.figure(figsize=(14, 8))

        # Determine the appropriate plot type based on the data
        data_counts = plot_data.groupby([alt_name_col, 'Criterion']).size()
        plot_type = "boxplot" if data_counts.max() > 1 else "scatter"

        if plot_type == "scatter":
            sns.scatterplot(
                data=plot_data,
                x=alt_name_col,
                y='Normalized Value',
                hue='Criterion',
                style='Criterion',
                s=100
            )
        elif plot_type == "boxplot":
            sns.boxplot(
                data=plot_data,
                x=alt_name_col,
                y='Normalized Value',
                hue='Criterion',
                fliersize=0
            )
        else:
            raise ValueError("Invalid plot_type. Expected 'scatter' or 'boxplot'.")

        # Customize plot labels
        plt.xlabel("Alternatives")
        y_label = "Normalized Criteria Values (Higher Scores are Better, 0–1 Scale)"
        if scale_to:
            y_label = f"Normalized Criteria Values (Higher Scores are Better, 1–{scale_to} Scale)"
        plt.ylabel(y_label)

        # Adjust the legend
        plt.legend(
            bbox_to_anchor=(0., 1.02, 1., .102), 
            loc='lower left', 
            ncol=4, 
            mode="expand", 
            borderaxespad=0., 
            edgecolor='black', 
            title='Criteria', 
            fontsize=12
        )
        plt.xticks(rotation=45)

        # Set y-ticks and grid for scaled data
        plt.grid(True, axis='y', linestyle='--', linewidth=0.7)

        plt.tight_layout()
        plt.show()



    def rescale_to_range(self, df, scale_to):
        """
        Rescale specified columns in a DataFrame to a 1–scale_to range.

        Parameters:
            df (DataFrame): DataFrame with normalized values to rescale.
            scale_to (int): Upper bound for the rescaling range.

        Returns:
            DataFrame: DataFrame with rescaled criteria columns.
        """
        for col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = 1 + (scale_to - 1) * (df[col] - min_val) / (max_val - min_val)
        return df
    
    def analyze_dominance_and_pareto(self, constraints=None, derived_columns=None):
        """
        Analyze dominance and Pareto optimality of alternatives in the decision matrix.

        Parameters:
            constraints (dict, optional): Constraints to filter the decision matrix before analysis.
            derived_columns (list, optional): Additional columns to include in the filtered DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the dominance and Pareto optimality results.
        """
        # Create a copy of the decision matrix
        dm_df = self.dm_df.copy()

        # Handle missing Group ID and Sample ID
        if 'Group ID' not in dm_df.columns:
            dm_df['Group ID'] = 'G1'
        if 'Sample ID' not in dm_df.columns:
            dm_df['Sample ID'] = 'S1'

        # Apply constraints if provided
        if constraints:
            dm_df, _ = filter_dataframe(dm_df, constraints, derived_columns=derived_columns)

        # Prepare the result DataFrame structure
        result_columns = self.alt_cols + ['Group ID', 'Sample ID', 'Dominated By', 'Dominance Rank', 'Pareto Optimal'] + self.crit_cols
        result_df = pd.DataFrame(columns=result_columns)

        # Iterate through unique combinations of Group ID and Sample ID
        for _, group_sample in dm_df[['Group ID', 'Sample ID']].drop_duplicates().iterrows():
            # Subset the data for the current group and sample
            subset = dm_df[
                (dm_df['Group ID'] == group_sample['Group ID']) &
                (dm_df['Sample ID'] == group_sample['Sample ID'])
            ]

            if subset.empty:
                continue  # Skip empty subsets

            # Extract only the criteria columns
            criteria_matrix = subset[self.crit_cols].to_numpy()

            # Initialize structures for dominance and Pareto tracking
            dominated_by = {i: [] for i in range(len(subset))}
            is_pareto_optimal = {i: True for i in range(len(subset))}

            # Perform dominance analysis
            for i, candidate in enumerate(criteria_matrix):
                for j, competitor in enumerate(criteria_matrix):
                    if i != j:
                        if np.all(competitor >= candidate) and np.any(competitor > candidate):
                            dominated_by[i].append(subset.iloc[j][self.alt_cols[0]])
                            is_pareto_optimal[i] = False

            # Calculate dominance rank (number of dominations)
            dominance_rank = {i: len(dominated_by[i]) for i in range(len(subset))}

            # Prepare results for the current group and sample
            temp_result = subset[self.alt_cols + ['Group ID', 'Sample ID'] + self.crit_cols].copy()
            temp_result['Dominated By'] = [dominated_by[i] for i in range(len(subset))]
            temp_result['Dominance Rank'] = [dominance_rank[i] for i in range(len(subset))]
            temp_result['Pareto Optimal'] = [is_pareto_optimal[i] for i in range(len(subset))]

            # Concatenate temp_result into result_df, handling empty DataFrame cases
            if result_df.empty:
                result_df = temp_result
            elif not temp_result.empty:
                result_df = pd.concat([result_df, temp_result], ignore_index=True)

        # Remove Group ID and Sample ID if not present in the original dataset
        if 'Group ID' not in self.dm_df.columns:
            result_df.drop(columns=['Group ID'], inplace=True)
        if 'Sample ID' not in self.dm_df.columns:
            result_df.drop(columns=['Sample ID'], inplace=True)

        return result_df


    

    def plot_pareto_frontier(self, criteria_x, criteria_y, constraints=None, derived_columns=None, show_table=False, group_id=None, sample_id=None):
        """
        Plot the Pareto frontier for two selected criteria and optionally display it as a table,
        with filtering based on group and sample IDs (if they exist).

        Parameters:
            criteria_x (str): The criterion to plot on the x-axis.
            criteria_y (str): The criterion to plot on the y-axis.
            constraints (dict, optional): Constraints to filter the data before analysis.
            derived_columns (list, optional): Additional columns to include in the filtered DataFrame.
            show_table (bool): If True, display a table of Pareto optimal alternatives.
            group_id (str, optional): Filter for a specific group ID.
            sample_id (str, optional): Filter for a specific sample ID.

        Returns:
            None: Displays a plot and optionally prints a table.
        """
        # Analyze dominance and Pareto optimality
        dominance_pareto_df = self.analyze_dominance_and_pareto(constraints=constraints, derived_columns=derived_columns)

        # Dynamically check for 'Group ID' and 'Sample ID'
        if 'Group ID' in dominance_pareto_df.columns and group_id:
            dominance_pareto_df = dominance_pareto_df[dominance_pareto_df['Group ID'] == group_id]
        if 'Sample ID' in dominance_pareto_df.columns and sample_id:
            dominance_pareto_df = dominance_pareto_df[dominance_pareto_df['Sample ID'] == sample_id]

        if dominance_pareto_df.empty:
            print("No data available for the specified group and sample IDs.")
            return

        # Filter Pareto optimal alternatives
        pareto_optimal_df = dominance_pareto_df[dominance_pareto_df['Pareto Optimal'] == True]

        # Check if criteria exist in the dataset
        missing_criteria = [c for c in [criteria_x, criteria_y] if c not in pareto_optimal_df.columns]
        if missing_criteria:
            raise ValueError(f"The following criteria are missing: {missing_criteria}. Ensure they are valid columns.")

        # Extract criteria values for Pareto optimal alternatives
        pareto_criteria_df = pareto_optimal_df[[criteria_x, criteria_y] + self.alt_cols]

        # Plot the Pareto frontier
        plt.figure(figsize=(10, 6))

        # Scatter plot for all alternatives in the filtered dataset
        group_sample_df = self.dm_df.copy()
        if 'Group ID' in group_sample_df.columns and group_id:
            group_sample_df = group_sample_df[group_sample_df['Group ID'] == group_id]
        if 'Sample ID' in group_sample_df.columns and sample_id:
            group_sample_df = group_sample_df[group_sample_df['Sample ID'] == sample_id]

        plt.scatter(group_sample_df[criteria_x], group_sample_df[criteria_y], color='gray', label='All Alternatives', alpha=0.6)

        # Highlight Pareto optimal alternatives
        plt.scatter(
            pareto_criteria_df[criteria_x],
            pareto_criteria_df[criteria_y],
            color='red',
            label='Pareto Optimal',
            s=100,
            edgecolor='black'
        )

        # Add text labels to the Pareto optimal points
        for _, row in pareto_criteria_df.iterrows():
            plt.text(row[criteria_x], row[criteria_y], row[self.alt_cols[0]], fontsize=10, ha='right', va='bottom')

        # Set plot labels and title
        plt.xlabel(criteria_x)
        plt.ylabel(criteria_y)

        # Adjust plot title based on group/sample availability
        group_sample_title = ""
        if 'Group ID' in self.dm_df.columns or 'Sample ID' in self.dm_df.columns:
            group_sample_title = f" (Group: {group_id or 'All'}, Sample: {sample_id or 'All'})"
        plt.title(f"Pareto Frontier for {criteria_y} vs {criteria_x}{group_sample_title}")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Optionally display a table of Pareto optimal alternatives
        if show_table:
            print("Pareto Optimal Alternatives:")
            print(tb.tabulate(pareto_criteria_df, headers='keys', tablefmt='pretty'))



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

# # Plot 
# def plot_crit_weights_sensitivity(imp_sens_df, xlabel):
#     # Plot the DataFrame as a stacked bar plot
#     ax = imp_sens_df.plot(kind='bar', stacked=True, figsize=(12, 6))

#     # Set the x-axis label
#     ax.set_xlabel(f'Weight of {xlabel}', fontsize=14)

#     # Set the y-axis label
#     ax.set_ylabel('Criteria Weight', fontsize=14)

#     # Set x-axis ticks to be in percentage format with no decimals
#     ax.set_xticklabels([f'{int(tick*100)}%' for tick in imp_sens_df.index], rotation=0, fontsize=12)

#     # Format the y-axis labels to be in percentage format with no decimals
#     ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

#     # Place the legend on the right side of the plot and set its title to "Criteria"
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Criteria', fontsize=12)

#     # Show the plot
#     plt.tight_layout()
#     plt.show()


# def plot_rank_sens_weights(ranks_imp_df, alt_tag, rank_method_name, xlabel, order_by='highest'):
#     # Pivot the DataFrame to make each alternative a column
#     plot_df = ranks_imp_df.pivot(index='Weight', columns=alt_tag, values=rank_method_name)

#     # Define a color map
#     color_map = cm.get_cmap('tab10', len(plot_df.columns))

#     # Create a dictionary that maps each column name to a specific color
#     color_dict = {col: color_map(i) for i, col in enumerate(plot_df.columns)}

#     # Reorder the columns according to the rank at the highest or lowest weights
#     if order_by == 'highest':
#         plot_df = plot_df[plot_df.iloc[-1].sort_values(ascending=False).index]
#         legend_loc = (1.05, 0.5)
#     elif order_by == 'lowest':
#         plot_df = plot_df[plot_df.iloc[0].sort_values(ascending=False).index]
#         legend_loc = (-0.3, 0.5)

#     # Plot the DataFrame with the color map
#     ax = plot_df.plot(kind='line', grid=True, figsize=(12, 6), color=[color_dict[col] for col in plot_df.columns])

#     # Set the x-axis label
#     ax.set_xlabel(f'Total weights of {xlabel}', fontsize=14)

#     # Set the y-axis label
#     ax.set_ylabel('Rank', fontsize=14)

#     # Format the x-axis labels to be in percentage format with no decimals
#     ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

#     # Set the y-axis limits
#     ax.set_ylim(0, plot_df.max().max() + 1)

#     # set the x-axis limits
#     ax.set_xlim(plot_df.index[0], plot_df.index[-1])

#     # Set the y-ticks to be from 1 to the maximum rank number
#     ax.yaxis.set_ticks(range(1, int(plot_df.max().max() + 2)))

#     # Enable the grid for each y-tick value
#     ax.yaxis.grid(True)

#     # Create a custom legend for the rank at the highest or lowest weights
#     lines = [mlines.Line2D([], [], color=color_dict[col], label=f'{col} ({int(plot_df.iloc[-1 if order_by == "highest" else 0, i])})') for i, col in enumerate(plot_df.columns)]
#     legend = plt.legend(handles=lines, bbox_to_anchor=legend_loc, loc='center left', borderaxespad=0., edgecolor='black', fontsize=14, title=f'Rank at {int(plot_df.index[-1 if order_by == "highest" else 0]*100)}%')

#     # Show the plot
#     plt.tight_layout()
#     plt.show()


# %%

    