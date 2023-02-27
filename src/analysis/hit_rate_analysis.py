import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def find_mdm2_motif(seq):
    # We're matching F**XX, where X is A,V,I,L,M,F,Y,W.
    return re.search(r"F\w\w(A|V|I|L|M|F|Y|W)(A|V|I|L|M|F|Y|W)", seq)


def seq_contains_mdm2_motif(seq):
    return find_mdm2_motif(seq) is not None


def find_12ca5_motif(seq):
    # We're matching DYA and DYS
    # Tho alternatively we maybe should match *D**DY(A|S)*
    return re.search(r"DY(S|A)", seq)


def seq_contains_12ca5_motif(seq):
    return find_12ca5_motif(seq) is not None


def sort_peptides_by_model_ranking(peptides, ranking):
    return [
        peptide
        for peptide, _ in sorted(
            zip(peptides, ranking),
            key=lambda pair: pair[1],
            reverse=True,
        )
    ]


# Stop at the top 500..

# Sort all of the peptides by their predicted confidence score * p-value confidence score.
# This presents a simple joint metric for peptide ranking.
def plot_ratio_by_ranking(
    peptides,
    mdm2_y_ranking,
    ca5_y_ranking,
    title,
    step_size=10,
    peptide_dataset_size=500,
    plot_theoretical_maximums=False,
    ylim=None,
):
    def compute_hit_ratios(
        sorted_peptides, hit_detection_function, step_size=step_size
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
            area_under_curve += hit_ratio * (
                min(i + step_size, peptide_dataset_size) - i
            )
        return hits_by_size, area_under_curve

    sorted_mdm2_peptides = sort_peptides_by_model_ranking(peptides, mdm2_y_ranking)
    mdm2_hit_ratios, mdm2_auc = compute_hit_ratios(
        sorted_mdm2_peptides,
        seq_contains_mdm2_motif,
    )
    mdm2_hits = [seq for seq in peptides if seq_contains_mdm2_motif(seq)]

    sorted_12ca5_peptides = sort_peptides_by_model_ranking(peptides, ca5_y_ranking)
    ca5_hit_ratios, ca5_auc = compute_hit_ratios(
        sorted_12ca5_peptides,
        seq_contains_12ca5_motif,
    )
    ca5_hits = [seq for seq in peptides if seq_contains_12ca5_motif(seq)]
    if plot_theoretical_maximums:
        dummy_optimal_mdm2_ranking = mdm2_hits + (
            [""] * (len(peptides) - len(mdm2_hits))
        )
        theoretical_best_mdm2_ratios, theoretical_best_mdm2_auc = compute_hit_ratios(
            dummy_optimal_mdm2_ranking,
            seq_contains_mdm2_motif,
        )

        dummy_optimal_ca5_ranking = ca5_hits + ([""] * (len(peptides) - len(ca5_hits)))
        theoretical_best_ca5_ratios, theoretical_best_ca5_auc = compute_hit_ratios(
            dummy_optimal_ca5_ranking,
            seq_contains_12ca5_motif,
        )

        # plt.plot(
        #     theoretical_best_mdm2_ratios[:, 1],
        #     theoretical_best_mdm2_ratios[:, 0],
        #     ":",
        #     label="Theoretical best discovery rate for MDM2",
        #     color="r",
        #     alpha=0.5,
        # )
        # plt.plot(
        #     theoretical_best_ca5_ratios[:, 1],
        #     theoretical_best_ca5_ratios[:, 0],
        #     ":",
        #     label="Theoretical best discovery rate for 12ca5",
        #     color="b",
        #     alpha=0.5,
        # )
        print("MDM2 area under curve: ", mdm2_auc)
        print(
            "normalized MDM2 area under curve: ", mdm2_auc / theoretical_best_mdm2_auc
        )
        print("12ca5 area under curve: ", ca5_auc)
        print("normalized 12ca5 area under curve: ", ca5_auc / theoretical_best_ca5_auc)

    # sns.set_style("darkgrid")
    if ylim is None:
        ylim = max(
            [
                # len(mdm2_hits),
                # len(ca5_hits),
                max(ca5_hit_ratios[:, 0]),
                max(mdm2_hit_ratios[:, 0]),
            ]
        )
    plt.ylim([0, ylim])
    plt.xlim([0, peptide_dataset_size])
    plt.plot(
        [0, peptide_dataset_size],
        [0, peptide_dataset_size],
        "--",
        color="#929591",
        label="Theoretical maximum",
    )

    plt.plot(
        ca5_hit_ratios[:, 1],
        ca5_hit_ratios[:, 0],
        label="12ca5 hits for motif DYX,\nwhere X is S,A",
        color="b",
        alpha=0.8,
    )

    plt.plot(
        mdm2_hit_ratios[:, 1],
        mdm2_hit_ratios[:, 0],
        label="MDM2 hits for motif F**XX,\nwhere X is A,V,I,L,M,F,Y,W",
        color="r",
        alpha=0.8,
    )
    # plt.fill_between(
    #     mdm2_hit_ratios[:, 1], y1=mdm2_hit_ratios[:, 0], y2=0, color="r", alpha=0.3
    # )

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.title(title)
    plt.ylabel("Number of Hits")
    plt.xlabel("Peptide Rank")

    plt.show()

    # print("mdm2 hits! ", mdm2_hits)
    # print("12ca5 hits! ", ca5_hits)
