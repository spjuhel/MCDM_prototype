import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Utility for consistent plot styling
def set_plot_style(figsize=(15, 8), fontsize=12):
    plt.rcParams.update({
        'figure.figsize': figsize,
        'axes.labelsize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'legend.fontsize': fontsize,
        'legend.title_fontsize': fontsize
    })
    sns.set_style("whitegrid")

def plot_grouped_criteria_weights(cat_crit_df):
    """
    Plots grouped criteria weights as a stacked bar plot.

    Parameters:
    - cat_crit_df (DataFrame): Categorized criteria DataFrame.

    Returns:
    None
    """
    set_plot_style()
    df = cat_crit_df.pivot(index='Category', columns='Criteria', values='Weight')
    ax = df.plot(kind='bar', stacked=True)

    cumulative_height = [0] * len(df)
    for i, p in enumerate(ax.patches):
        bar_index = i % len(df)
        cumulative_height[bar_index] += p.get_height()
        if p.get_height() > 0:
            ax.annotate(
                f'{p.get_height():.2f}',
                (p.get_x() + p.get_width() / 2., cumulative_height[bar_index] - p.get_height() / 2),
                ha='center', va='center', fontsize=10
            )

    ax.set_xlabel('Criteria categories', fontsize=12)
    ax.set_xticklabels([label[:10] for label in df.index], rotation=45, ha='right')
    plt.legend(
        bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=4, mode="expand",
        borderaxespad=0., edgecolor='black', title='Criteria', fontsize=12
    )
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()

def plot_individual_criteria_weights(crit_df):
    """
    Plots individual criteria weights as a simple bar plot.

    Parameters:
    - crit_df (DataFrame): Criteria DataFrame.

    Returns:
    None
    """
    set_plot_style()
    ax = crit_df.plot(x='Criteria', y='Weight', kind='bar', legend=False)
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(
                f'{p.get_height():.2f}',
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', fontsize=10
            )

    ax.set_xlabel('Criteria', fontsize=12)
    ax.set_xticklabels([label[:10] for label in crit_df['Criteria']], rotation=45, ha='right')
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()

# Make a utiltiy function to compare attributes between the original and refactored classes
def compare_attributes(obj1, obj2):
    """
    Compares attributes between two objects and prints differences.

    Parameters:
    - obj1: The first object to compare.
    - obj2: The second object to compare.

    Returns:
    None. Prints results directly.
    """
    obj1_attrs = vars(obj1)
    obj2_attrs = vars(obj2)
    
    differences = []
    for key in obj1_attrs:
        if key in obj2_attrs:
            value1 = obj1_attrs[key]
            value2 = obj2_attrs[key]
            
            # Handle DataFrame comparison
            if isinstance(value1, pd.DataFrame) and isinstance(value2, pd.DataFrame):
                if not value1.equals(value2):
                    differences.append(f"DataFrame mismatch in attribute '{key}'")
            
            # Handle Series comparison
            elif isinstance(value1, pd.Series) and isinstance(value2, pd.Series):
                if not value1.equals(value2):
                    differences.append(f"Series mismatch in attribute '{key}'")
            
            # Handle list comparison
            elif isinstance(value1, list) and isinstance(value2, list):
                if value1 != value2:
                    differences.append(f"List mismatch in attribute '{key}'")
            
            # Handle dictionary comparison
            elif isinstance(value1, dict) and isinstance(value2, dict):
                for subkey in set(value1.keys()).union(value2.keys()):
                    if subkey not in value1:
                        differences.append(f"Key '{subkey}' missing in dictionary attribute '{key}' for obj1")
                    elif subkey not in value2:
                        differences.append(f"Key '{subkey}' missing in dictionary attribute '{key}' for obj2")
                    else:
                        val1 = value1[subkey]
                        val2 = value2[subkey]
                        if isinstance(val1, pd.DataFrame) and isinstance(val2, pd.DataFrame):
                            if not val1.equals(val2):
                                differences.append(f"DataFrame mismatch in dictionary attribute '{key}' for subkey '{subkey}'")
                        elif isinstance(val1, pd.Series) and isinstance(val2, pd.Series):
                            if not val1.equals(val2):
                                differences.append(f"Series mismatch in dictionary attribute '{key}' for subkey '{subkey}'")
                        elif val1 != val2:
                            differences.append(f"Value mismatch in dictionary attribute '{key}' for subkey '{subkey}': {val1} != {val2}")
            
            # Handle None values
            elif value1 is None and value2 is None:
                continue  # Both are None, so they match
            
            # Mismatched types or fallback for other types
            elif type(value1) != type(value2):
                differences.append(f"Type mismatch for attribute '{key}': {type(value1)} != {type(value2)}")
            else:
                if value1 != value2:
                    differences.append(f"Value mismatch in attribute '{key}': {value1} != {value2}")
        
        else:
            differences.append(f"Attribute '{key}' missing in obj2")
    
    for key in obj2_attrs:
        if key not in obj1_attrs:
            differences.append(f"Attribute '{key}' missing in obj1")
    
    # Print results
    if not differences:
        print("All attributes match between the original and refactored versions.")
    else:
        print("Differences found in attributes:")
        for diff in differences:
            print(diff)