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
