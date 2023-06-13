import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pdb

# import scienceplots
# plt.style.use([])# 'notebook']) #,'ieee'])


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
    save_file='mdm2_hit_rankings.csv',
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

            hit_ratio = len(hit_indices) / len(top_peptides)
            hits_by_size = np.concatenate(
                (hits_by_size, np.array([(len(hit_indices), len(top_peptides))])),
                axis=0,
            )
            area_under_curve += len(hit_indices) * step_size
            # area_under_curve += hit_ratio * (
            #     min(i + step_size, peptide_dataset_size) - i
            # )
        return hits_by_size, area_under_curve

    # Plot theoretical hits:
    all_mdm2_hits = [pep for pep in peptides if hit_rate_func(pep)]
    dummy_optimal_mdm2_ranking = all_mdm2_hits + (
        [""] * (len(peptides) - len(all_mdm2_hits))
    )
    theoretical_best_mdm2_ratios, theoretical_best_mdm2_auc = compute_hit_ratios(
        dummy_optimal_mdm2_ranking,
        hit_rate_func,
    )
    result_df = pd.DataFrame(
        theoretical_best_mdm2_ratios,
        columns=["# Peptide", "Theoretical Best Ranking"],
    )
    plt.plot(
        theoretical_best_mdm2_ratios[:, 1],
        theoretical_best_mdm2_ratios[:, 0],
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
        sorted_mdm2_peptides = sort_peptides_by_model_ranking(peptides, y_ranking)
        mdm2_hit_ratios, mdm2_auc = compute_hit_ratios(
            sorted_mdm2_peptides,
            hit_rate_func,
        )
        mdm2_hits = [seq for seq in peptides if hit_rate_func(seq)]
        normalized_auc = mdm2_auc / theoretical_best_mdm2_auc
        result_df[label] = mdm2_hit_ratios[:, 0]

        plt.plot(
            mdm2_hit_ratios[:, 1],
            mdm2_hit_ratios[:, 0],
            label=label +  "\nNormalized Hit Rate AUC: {0:.3f}".format(normalized_auc),
            markevery=markers_on,
            color=color,
            alpha=0.8,
            marker=markers[idx],
        )
        if normalized_auc > best_curve_auc:
            best_curve_auc = normalized_auc
            best_hit_ratios = mdm2_hit_ratios

    if ylim is None:
        ylim = max(
            [
                600
                # 1.2 * max(best_hit_ratios[:, 0]),
                # 500
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

    # print("mdm2 hits! ", mdm2_hits)
    # print("12ca5 hits! ", ca5_hits)

    # dummy_optimal_ca5_ranking = ca5_hits + ([""] * (len(peptides) - len(ca5_hits)))
    # theoretical_best_ca5_ratios, theoretical_best_ca5_auc = compute_hit_ratios(
    #     dummy_optimal_ca5_ranking,
    #     seq_contains_12ca5_motif,
    # )

    # plt.plot(
    #     theoretical_best_ca5_ratios[:, 1],
    #     theoretical_best_ca5_ratios[:, 0],
    #     ":",
    #     label="Theoretical best discovery rate for 12ca5",
    #     color="b",
    #     alpha=0.5,
    # )
    # print("12ca5 area under curve: ", ca5_auc)
    # print("normalized 12ca5 area under curve: ", ca5_auc / theoretical_best_ca5_auc)

    # sns.set_style("darkgrid")
    # plt.plot(
    #     [0, peptide_dataset_size],
    #     [0, peptide_dataset_size],
    #     "--",
    #     color="#929591",
    #     label="Theoretical maximum",
    # )

    # plt.plot(
    #     ca5_hit_ratios[:, 1],
    #     ca5_hit_ratios[:, 0],
    #     label="12ca5 hits for motif DYX,\nwhere X is S,A",
    #     color="b",
    #     alpha=0.8,
    # )
