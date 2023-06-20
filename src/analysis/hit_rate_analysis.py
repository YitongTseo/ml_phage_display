import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from analysis.scatter_plots import (
    plot_relations_in_3D,
    plot_fancy_hexbin_relations,
    plot_relations,
)
from tensorflow import keras
from functools import partial
from utils.utils import seed_everything


def find_mdm2_motif(seq):
    """
    We're matching F**$$, where $ = F,W,L,I,V,Y

    https://www.nature.com/articles/s42004-022-00737-w,
    https://www.pnas.org/doi/10.1073/pnas.1303002110,
    https://pubs.acs.org/doi/10.1021/bi060309g,
    https://pubs.acs.org/doi/10.1021/ja0693587,
    https://www.jbc.org/article/S0021-9258(19)64923-9/fulltext
    """
    # TODO: rerun...
    return re.search(r"F\w\w(F|W|L|I|V|Y)(F|W|L|I|V|Y)", seq)


def seq_contains_mdm2_motif(seq):
    return (find_mdm2_motif(seq) is not None) and (find_12ca5_motif(seq) is None)


def find_12ca5_motif(seq):
    # We're matching DYA and DYS
    return re.search(r"DY(S|A)", seq)


def seq_contains_12ca5_motif(seq):
    return (find_12ca5_motif(seq) is not None) and (find_mdm2_motif(seq) is None)


def sort_peptides_by_model_ranking(peptides, ranking):
    return [
        peptide
        for peptide, _ in sorted(
            zip(peptides, ranking),
            key=lambda pair: pair[1],
            reverse=True,
        )
    ]


# Sort all of the peptides by their predicted confidence score * p-value confidence score.
# This presents a simple joint metric for peptide ranking.
def plot_ratio_by_ranking(
    peptides,
    y_rankings,
    title,
    hit_rate_func=seq_contains_mdm2_motif,
    step_size=10,
    peptide_dataset_size=500,
    ylim=None,
    save_file="hit_rankings.csv",
    plot=True,
):
    def compute_hit_ratios(
        sorted_peptides,
        hit_detection_function,
        step_size=step_size,
        peptide_dataset_size=peptide_dataset_size,
    ):
        hits_by_size = np.array([])
        hits_by_size.shape = (0, 2)
        area_under_curve = 0
        for i in range(0, peptide_dataset_size, step_size):
            top_peptides = sorted_peptides[0 : min(i + step_size, peptide_dataset_size)]
            hit_indices = np.where(
                [hit_detection_function(seq) for seq in top_peptides]
            )[0]
            hits_by_size = np.concatenate(
                (hits_by_size, np.array([(len(hit_indices), len(top_peptides))])),
                axis=0,
            )
            area_under_curve += len(hit_indices) * step_size
        return hits_by_size, area_under_curve

    # Plot theoretical hits:
    all_hits = [pep for pep in peptides if hit_rate_func(pep)]
    dummy_optimal_ranking = all_hits + ([""] * (len(peptides) - len(all_hits)))
    theoretical_best_ratios, theoretical_best_auc = compute_hit_ratios(
        dummy_optimal_ranking,
        hit_rate_func,
    )
    result_df = pd.DataFrame(
        theoretical_best_ratios,
        columns=["# Peptide", "Theoretical Best Ranking"],
    )
    plt.plot(
        theoretical_best_ratios[:, 1],
        theoretical_best_ratios[:, 0],
        "--",
        label="Theoretical Maximum",
        color="#929591",
        alpha=0.5,
    )

    # Plot Supplied Rankings:
    best_curve_auc = 0
    best_hit_ratios = None
    markers_on = list(range(int(peptide_dataset_size / step_size)))
    markers = ["s", "*", "o", "^", "D"]

    for idx, (y_ranking, label, color) in enumerate(y_rankings):
        sorted_peptides = sort_peptides_by_model_ranking(peptides, y_ranking)
        hit_ratios, _auc = compute_hit_ratios(
            sorted_peptides,
            hit_rate_func,
        )
        hits = [seq for seq in peptides if hit_rate_func(seq)]
        normalized_auc = _auc / theoretical_best_auc
        result_df[label] = hit_ratios[:, 0]

        plt.plot(
            hit_ratios[:, 1],
            hit_ratios[:, 0],
            label=label + "\nNormalized Hit Rate AUC: {0:.3f}".format(normalized_auc),
            markevery=markers_on,
            color=color,
            alpha=0.8,
            marker=markers[idx],
        )
        if normalized_auc > best_curve_auc:
            best_curve_auc = normalized_auc
            best_hit_ratios = hit_ratios

    if ylim is None:
        ylim = min(
            [
                ylim if ylim is not None else np.inf,
                1.2 * max(best_hit_ratios[:, 0]),
            ]
        )
    if save_file is not None:
        result_df.to_csv(save_file)
    if plot:
        ax = plt.gca()

        plt.ylim([0, ylim])
        plt.xlim([0, peptide_dataset_size + 3])
        ax.set_yticks([0, ylim])
        ax.set_xticks([0, peptide_dataset_size])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
        plt.title(title, family="Arial")
        plt.ylabel("Number of Hits", family="Arial")
        plt.xlabel("Peptide Rank", family="Arial")

        plt.show()

    return best_curve_auc


def benchmark_cross_validated_hit_rate(
    cross_validation_results,
    y_raw,
    peptides,
    proxy_ranking_lambda,
    top_k_size,
    motif_dectection_func,
    calculate_proxy_uncertainty=False,
    plot_x_idx=1,
    plot_y_idx=0,
    plot_confusion_results=False,
):
    seed_everything(0)
    y_pred = np.vstack(result.y_pred_rescaled for result in cross_validation_results)
    y_true = np.vstack(result.y_test for result in cross_validation_results)
    # Check that the cross folds experiment returns our y_true in the same order as before
    assert (y_true == y_raw).all()

    all_positives = np.array(
        [1.0 if motif_dectection_func(pep) else 0.0 for pep in peptides]
    )
    if calculate_proxy_uncertainty:
        # Calculate with dropout on as proxy uncertainty
        pred_100_fold = []
        for result in cross_validation_results:
            pred_100_fold.append(
                np.array(
                    [
                        result.trained_model(result.X_test, training=True)
                        for _ in range(100)
                    ]
                )
            )
        pred_100_fold = np.concatenate(pred_100_fold, axis=1)

        mean = np.mean(pred_100_fold, axis=0)
        variance = np.std(pred_100_fold, axis=0)
        uncertainty = np.mean(variance, axis=1)

        _ordering = [proxy_ranking_lambda(pred) for pred in mean]
        y_pred = mean
        if plot_confusion_results:
            plot_relations_in_3D(
                plot_x_idx,
                plot_y_idx,
                datapoints=mean,
                title="Predicted Hits Coloring",
                ordering=_ordering,
                all_positives=all_positives,
                uncertainty=uncertainty,
            )
    else:
        _ordering = [proxy_ranking_lambda(pred) for pred in y_pred]
        if plot_confusion_results:
            plot_relations(
                plot_x_idx,
                plot_y_idx,
                datapoints=y_pred,
                ordering=_ordering,
                all_positives=all_positives,
                kind="scatter",
            )
    plot_fancy_hexbin_relations(
        plot_x_idx,
        plot_y_idx,
        datapoints=y_pred,
        ordering=_ordering,
        all_positives=None,
        line_color="#F94040",
        vals=[
            "Predicted -log(P-value)",
            "Predicted log(Fold Change)",
            "Predicted Enrichment Ratio",
        ],
        top_k=top_k_size,
    )
    plot_fancy_hexbin_relations(
        plot_x_idx,
        plot_y_idx,
        datapoints=y_pred,
        ordering=None,
        all_positives=all_positives,
        line_color="#F94040",
        vals=[
            "Predicted -log(P-value)",
            "Predicted log(Fold Change)",
            "Predicted Enrichment Ratio",
        ],
        top_k=top_k_size,
    )
    return _ordering, y_pred
