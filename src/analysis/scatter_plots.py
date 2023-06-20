import pathlib
import sys

HOME_DIRECTORY = pathlib.Path().absolute()
sys.path.append(str(HOME_DIRECTORY) + "/src")

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import pdb
import matplotlib
import copy
from matplotlib.pyplot import figure


TRUE_POSITIVE = (0, 1, 0, 0.5)
FALSE_POSITIVE = (0, 0, 1, 0.5)
FALSE_NEGATIVE = (1, 0, 0, 0.5)
TRUE_NEGATIVE = (0.8, 0.8, 0.8, 0.03)


def plot_relations_in_plotly(
    x_idx, y_idx, peptides, datapoints, vals=["Pval", "FC", "ER"]
):
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


def plot_fancy_hexbin_relations(
    x_idx,
    y_idx,
    datapoints,
    ordering,
    all_positives,
    vals=["-log(P-value)", "log(Fold Change)", "Enrichment Ratio"],
    line_color="black",
    top_k=500,
    gridsize=30,
    title=None,
    plot_labels=True,
    line_weight=8,
):
    # fig = plt.figure(figsize=(6,6))

    figure(figsize=(12, 12), dpi=80)
    font = {"family": "Arial", "weight": "normal", "size": 22}

    matplotlib.rc("font", **font)

    if plot_labels:
        plt.xlabel(vals[x_idx])
        plt.ylabel(vals[y_idx])
    if title is None:
        plt.title(vals[x_idx] + " vs " + vals[y_idx])
    else:
        plt.title(title)

    cmap = copy.copy(matplotlib.colormaps["Greys"])  # copy the default cmap
    cmap.set_bad((1, 1, 1))

    # original plot
    extent = (
        datapoints[:, x_idx].min() - (0.1 * datapoints[:, x_idx].var()),
        datapoints[:, x_idx].max() + (0.3 * datapoints[:, x_idx].var()),
        datapoints[:, y_idx].min() - (0.1 * datapoints[:, y_idx].var()),
        datapoints[:, y_idx].max() + (0.3 * datapoints[:, y_idx].var()),
    )
    print(extent)
    cfg = dict(
        x=datapoints[:, x_idx],
        y=datapoints[:, y_idx],
        cmap=cmap,
        norm=matplotlib.colors.LogNorm(clip=False),
        gridsize=gridsize,
        extent=extent,
    )
    h = plt.hexbin(ec="white", lw=1, zorder=-5, **cfg)

    if all_positives is not None:
        plt.scatter(
            x=datapoints[all_positives.astype(bool)][:, x_idx],
            y=datapoints[all_positives.astype(bool)][:, y_idx],
            c="forestgreen",
            marker="x",
            alpha=0.5,
        )

    if ordering is not None:
        top_k_mask = ordering >= np.partition(ordering, kth=-top_k)[-top_k]
        top_k_datapoints = datapoints[top_k_mask]

        plt.hexbin(
            x=top_k_datapoints[:, x_idx],
            y=top_k_datapoints[:, y_idx],
            cmap=cmap,
            norm=matplotlib.colors.LogNorm(clip=False),
            gridsize=gridsize,
            extent=extent,
            ec=line_color,
            lw=line_weight,
            zorder=-3,
            mincnt=1,
        )
        plt.hexbin(
            x=top_k_datapoints[:, x_idx],
            y=top_k_datapoints[:, y_idx],
            cmap=cmap,
            norm=matplotlib.colors.LogNorm(clip=False),
            gridsize=gridsize,
            extent=extent,
            ec="white",
            lw=0,
            zorder=-2,
            mincnt=1,
        )
        plt.hexbin(
            alpha=0.5,
            cmap=matplotlib.colors.ListedColormap([line_color]),
            x=top_k_datapoints[:, x_idx],
            y=top_k_datapoints[:, y_idx],
            gridsize=gridsize,
            extent=extent,
            ec="white",
            lw=0,
            zorder=-2,
            mincnt=1,
        )

        plt.xlim(extent[0], extent[1])  # required as second call of plt.hexbin()
        plt.ylim(extent[2], extent[3])  # strangely affects the limits ...

    plt.show()


def plot_relations(
    x_idx,
    y_idx,
    datapoints,
    ordering,
    all_positives,
    kind="hex",
    vals=["Pval", "FC", "ER"],
    top_k=500,
):
    g = sns.jointplot(x=datapoints[:, x_idx], y=datapoints[:, y_idx], kind=kind)
    plt.xlabel(vals[x_idx])
    plt.ylabel(vals[y_idx])
    plt.title(vals[x_idx] + " vs " + vals[y_idx])

    if ordering is not None and all_positives is not None:
        g.ax_joint.cla()
        top_k_mask = ordering >= np.partition(ordering, kth=-top_k)[-top_k]
        colors = []
        for datapoint, is_true, is_in_top_k in zip(
            datapoints, all_positives, top_k_mask
        ):
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

        g.ax_joint.legend(
            handles=[
                mpatches.Patch(
                    color=TRUE_POSITIVE, label="True Positive (peptide hit in top 500)"
                ),
                mpatches.Patch(
                    color=FALSE_POSITIVE,
                    label="False Positive (peptide miss in top 500)",
                ),
                mpatches.Patch(
                    color=FALSE_NEGATIVE,
                    label="False Negative (peptide hit not in top 500)",
                ),
                mpatches.Patch(
                    color=TRUE_NEGATIVE,
                    label="True Negative (peptide miss not in top 500)",
                ),
            ],
            loc="best",
        )
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

        ax.legend(
            handles=[
                mpatches.Patch(
                    color=TRUE_POSITIVE, label="True Positive (peptide hit in top 500)"
                ),
                mpatches.Patch(
                    color=FALSE_POSITIVE,
                    label="False Positive (peptide miss in top 500)",
                ),
                mpatches.Patch(
                    color=FALSE_NEGATIVE,
                    label="False Negative (peptide hit not in top 500)",
                ),
                mpatches.Patch(
                    color=TRUE_NEGATIVE,
                    label="True Negative (peptide miss not in top 500)",
                ),
            ]
        )
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
    plt.show()


def show_volcano(
    y,
    title,
    x_label,
    y_label,
    mdm2_ordering=None,
    ca5_ordering=None,
    top_k_size=500,
):
    sns.set_palette(sns.color_palette("plasma"))
    p = sns.jointplot(
        x=y[:, 1],
        y=y[:, 0],
        color="grey",
        alpha=0.4,
        marginal_kws=dict(bins=50, fill=False),
    )
    if mdm2_ordering is not None:
        top_mdm2_mask = (
            mdm2_ordering >= np.partition(mdm2_ordering, kth=-top_k_size)[-top_k_size]
        )
        plt.scatter(
            x=y[top_mdm2_mask][:, 1],
            y=y[top_mdm2_mask][:, 0],
            color="#F94040",
            alpha=0.8,
        )
    if ca5_ordering is not None:
        top_ca5_mask = (
            ca5_ordering >= np.partition(ca5_ordering, kth=-top_k_size)[-top_k_size]
        )
        plt.scatter(
            x=y[top_ca5_mask][:, 1],
            y=y[top_ca5_mask][:, 0],
            color="blue",
            alpha=0.8,
        )

    p.fig.suptitle(title)
    p.set_axis_labels(x_label, y_label)
    p.fig.subplots_adjust(top=0.95)  # Reduce plot to make room
