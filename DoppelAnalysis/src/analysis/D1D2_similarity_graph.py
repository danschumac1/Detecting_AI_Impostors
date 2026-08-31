"""
2026-08-30
Author: Dan Schumacher

python ./src/analysis/D1D2_similarity_graph
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter


def main():

    # Values from paired semantic similarity analysis
    data = {
        'Metric': ['BERT**', 'LIWC**', 'METEOR*', 'ROUGE-L'],
        'Day 1': [0.727, 0.770, 0.131, 0.115],
        'Day 2': [0.790, 0.864, 0.187, 0.147]
    }

    df = pd.DataFrame(data)

    df_melted = df.melt(
        id_vars='Metric',
        var_name='Day',
        value_name='Similarity'
    )

    palette = {
        'Day 1': '#003f5c',
        'Day 2': '#ffa600'
    }

    sns.set_context("talk")
    sns.set_style("white")

    plt.figure(figsize=(8, 6))

    ax = sns.barplot(
        data=df_melted,
        x='Metric',
        y='Similarity',
        hue='Day',
        palette=palette
    )

    # Value labels
    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[
                f'{h:.3f}'.lstrip('0')
                for h in container.datavalues
            ],
            padding=3,
            fontsize=14
        )

    ax.set_ylim(0, 0.90)

    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda y, _: f'{y:.1f}'.lstrip('0')
        )
    )

    ax.set_ylabel(
        'Average Similarity',
        fontsize=18
    )
    ax.set_xlabel('')

    ax.tick_params(
        axis='both',
        labelsize=18
    )


    ax.legend(
        title='',
        fontsize=18,
        loc='upper right'
    )

    # Show only left and bottom axis lines
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    plt.tight_layout()

    plt.savefig(
        './figures/D1D2_similarity_graph.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.savefig(
        './figures/D1D2_similarity_graph.pdf',
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()

    print(f"Graph saved to ./figures/D1D2_similarity_graph")
if __name__ == "__main__":
    main()