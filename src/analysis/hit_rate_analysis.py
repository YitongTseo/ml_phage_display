import re
import matplotlib.pyplot as plt
import numpy as np

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
    
def sort_peptides_by_model_ranking(peptides_test, y_pred_raw):
    return [
        peptide
        for peptide, _ in sorted(
            zip(peptides_test, y_pred_raw[:, 1] * y_pred_raw[:, 0]),
            key=lambda pair: pair[1],
            reverse=True,
        )
    ]

# TODO(Yitong & Yehlin): we might want to do something a little more sophisticated here
# Where we only take the peptides which are in the partition of the UMAP we care about...

# Sort all of the peptides by their predicted confidence score * p-value confidence score.
# This presents a simple joint metric for peptide ranking.
def plot_ratio_by_ranking(
    peptides_test, mdm2_y_pred_raw, ca5_y_pred_raw, title, step_size=100, plot_theoretical_maximums=True
):
    def compute_hit_ratios(
        sorted_peptides, hit_detection_function, step_size=step_size
    ):
        hit_ratios = np.array([])
        hit_ratios.shape = (0, 2)
        top_hits = None
        area_under_curve = 0
        for i in range(0, len(sorted_peptides), step_size):
            top_peptides = sorted_peptides[0 : min(i + step_size, len(sorted_peptides))]
            hit_indices = np.where(
                [hit_detection_function(seq) for seq in top_peptides]
            )[0]
            top_hits = [
                peptide
                for (idx, peptide) in enumerate(top_peptides)
                if idx in hit_indices
            ]
            hit_ratio = len(hit_indices) / len(top_peptides)
            dataset_ratio = len(top_peptides) / len(sorted_peptides)
            hit_ratios = np.concatenate(
                (hit_ratios, np.array([(hit_ratio, dataset_ratio)])), axis=0
            )
            area_under_curve += (hit_ratio * (min(i + step_size, len(sorted_peptides)) - i))
        return hit_ratios, top_hits, area_under_curve

    sorted_mdm2_peptides = sort_peptides_by_model_ranking(
        peptides_test, mdm2_y_pred_raw
    )
    mdm2_hit_ratios, mdm2_hits, mdm2_auc = compute_hit_ratios(
        sorted_mdm2_peptides, seq_contains_mdm2_motif
    )

    sorted_12ca5_peptides = sort_peptides_by_model_ranking(
        peptides_test, ca5_y_pred_raw
    )
    ca5_hit_ratios, ca5_hits, ca5_auc = compute_hit_ratios(
        sorted_12ca5_peptides, seq_contains_12ca5_motif
    )
    if plot_theoretical_maximums:
        # def get_theoretical_curve(total_num_hits, total_peptides=len(peptides_test), step_size=step_size):
        #     theoretical_best_ratios = np.array([])
        #     theoretical_best_ratios.shape = (0, 2)
        #     for i in range(0, total_peptides, step_size):
        #         set_size = min(i + step_size, total_peptides)
        #         theoretical_best_ratio = min(total_num_hits / set_size, 1)
        #         dataset_ratio = set_size / total_peptides
        #         theoretical_best_ratios = np.concatenate(
        #             (theoretical_best_ratios, np.array([(theoretical_best_ratio, dataset_ratio)])), axis=0
        #         )
        #     return theoretical_best_ratios
        # theoretical_best_mdm2 = get_theoretical_curve(len(mdm2_hits))
        # theoretical_best_12ca5 = get_theoretical_curve(len(ca5_hits))
        dummy_optimal_mdm2_ranking = mdm2_hits + ([''] * (len(peptides_test)- len(mdm2_hits)))
        theoretical_best_mdm2_ratios, _, theoretical_best_mdm2_auc = compute_hit_ratios(
            dummy_optimal_mdm2_ranking, seq_contains_mdm2_motif
        )

        dummy_optimal_ca5_ranking = ca5_hits + ([''] * (len(peptides_test)- len(ca5_hits)))
        theoretical_best_ca5_ratios, _, theoretical_best_ca5_auc = compute_hit_ratios(
            dummy_optimal_ca5_ranking, seq_contains_12ca5_motif
        )

        plt.plot(theoretical_best_mdm2_ratios[:, 1], theoretical_best_mdm2_ratios[:, 0], ':', label='Theoretical max MDM2')
        plt.plot(theoretical_best_ca5_ratios[:, 1], theoretical_best_ca5_ratios[:, 0], ':', label='Theoretical max 12ca5')

        print('normalized MDM2 area under curve: ', mdm2_auc / theoretical_best_mdm2_auc)
        print('normalized 12ca5 area under curve: ', ca5_auc / theoretical_best_ca5_auc)


    plt.plot(mdm2_hit_ratios[:, 1], mdm2_hit_ratios[:, 0], label='MDM2 hits for motif F**XX,\nwhere X is A,V,I,L,M,F,Y,W')
    plt.plot(ca5_hit_ratios[:, 1], ca5_hit_ratios[:, 0], label='12ca5 hits for motif DYX,\nwhere X is S,A')
    plt.legend()
    plt.title(title)
    plt.ylabel("Hit Rate Ratio")
    plt.xlabel("Ranked Peptide Set Size Ratio to Full Test Set Size")
    plt.show()

    print('mdm2 hits! ', mdm2_hits)
    print('12ca5 hits! ', ca5_hits)