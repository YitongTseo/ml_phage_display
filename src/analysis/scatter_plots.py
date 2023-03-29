import pathlib
import sys
HOME_DIRECTORY = pathlib.Path().absolute()
sys.path.append(str(HOME_DIRECTORY) + '/src')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import pdb

TRUE_POSITIVE = (0, 1, 0, 0.5)
FALSE_POSITIVE = (0, 0, 1, 0.5)
FALSE_NEGATIVE = (1, 0, 0, 0.5)
TRUE_NEGATIVE = (0.8, 0.8, 0.8, 0.3)

def plot_relations_in_plotly(x_idx, y_idx, peptides, datapoints, vals=["Pval", "FC", "ER"]):
    fig = go.Figure(
        [
            go.Scatter(
                x=datapoints[:, x_idx],
                y=datapoints[:, y_idx],
                text=peptides,
                mode="markers",
            )
        ]
    )
    fig.show()


def plot_relations(
    x_idx, y_idx, datapoints, ordering, all_positives, kind="hex", vals=["Pval", "FC", "ER"], top_k=500
):
    g = sns.jointplot(x=datapoints[:, x_idx], y=datapoints[:, y_idx], kind=kind)
    plt.xlabel(vals[x_idx])
    plt.ylabel(vals[y_idx])
    plt.title(vals[x_idx] + ' vs ' + vals[y_idx])
    
    if ordering is not None and all_positives is not None:
        g.ax_joint.cla()
        top_k_mask = ordering >= np.partition(ordering, kth=-top_k)[-top_k]
        colors = []
        for datapoint, is_true, is_in_top_k in zip(datapoints, all_positives, top_k_mask):
            if is_true and is_in_top_k:
                colors.append(TRUE_POSITIVE) 

            if is_true and not is_in_top_k:
                colors.append(FALSE_NEGATIVE) 

            if not is_true and is_in_top_k:
                colors.append(FALSE_POSITIVE) 

            if not is_true and not is_in_top_k:
                colors.append(TRUE_NEGATIVE)

        plt.scatter(
            x=datapoints[:, x_idx],
            y=datapoints[:, y_idx],
            c=colors,
        )
        
        g.ax_joint.legend(handles=[
            mpatches.Patch(color=TRUE_POSITIVE, label='True Positive (peptide hit in top 500)'),
            mpatches.Patch(color=FALSE_POSITIVE, label='False Positive (peptide miss in top 500)'),
            mpatches.Patch(color=FALSE_NEGATIVE, label='False Negative (peptide hit not in top 500)'),
            mpatches.Patch(color=TRUE_NEGATIVE, label='True Negative (peptide miss not in top 500)'),
        ],loc='best')
    plt.show()


def plot_relations_in_3D(
    x_idx,
    y_idx,
    uncertainty,
    datapoints,
    ordering,
    all_positives,
    title="",
    vals=["Pval", "FC", "ER"],
    top_k=500,
):
    sns.set_style("whitegrid", {"axes.grid": False})
    fig = plt.figure(figsize=(6, 6))

    ax = Axes3D(fig)
    if ordering is not None and all_positives is not None:
        top_k_mask = ordering >= np.partition(ordering, kth=-top_k)[-top_k]
        colors = []
        for is_true, is_in_top_k in zip(all_positives, top_k_mask):
            if is_true and is_in_top_k:
                colors.append(TRUE_POSITIVE) 
            
            if is_true and not is_in_top_k:
                colors.append(FALSE_NEGATIVE) 

            if not is_true and is_in_top_k:
                colors.append(FALSE_POSITIVE) 

            if not is_true and not is_in_top_k:
                colors.append(TRUE_NEGATIVE)

        ax.legend(handles=[
            mpatches.Patch(color=TRUE_POSITIVE, label='True Positive (peptide hit in top 500)'),
            mpatches.Patch(color=FALSE_POSITIVE, label='False Positive (peptide miss in top 500)'),
            mpatches.Patch(color=FALSE_NEGATIVE, label='False Negative (peptide hit not in top 500)'),
            mpatches.Patch(color=TRUE_NEGATIVE, label='True Negative (peptide miss not in top 500)'),
        ])
    else:
        colors = uncertainty

    ax.scatter(
        xs=datapoints[:, x_idx],
        ys=datapoints[:, y_idx],
        zs=uncertainty,
        c=colors,
        marker="o",
    )
    ax.set_xlabel("Pred " + vals[x_idx])
    ax.set_ylabel("Pred " + vals[y_idx])
    ax.set_zlabel("Uncertainty")

    plt.title(title)
    # ax.view_init(elev=10.0, azim=90)
    plt.show()