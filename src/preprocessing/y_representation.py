from functools import reduce, partial
import pandas as pd
import numpy as np
import scipy as sp
import pdb

REPLICATE_IDXS = (" 1", " 2", " 3")
ALL_PROTEINS = ["12ca5", "MDM2"]


def formulate_two_channel_regression_labels(lib, protein_of_interest, other_protein):
    lib["poi_log_fold"] = calculate_log_fold_across_proteins(
        lib, protein_of_interest, [other_protein]
    )
    lib["poi_log_pvalue"] = calculate_log_p_value(
        lib, protein_of_interest, other_protein
    )
    return np.stack(
        (lib["poi_log_pvalue"].to_numpy(), lib["poi_log_fold"].to_numpy()), axis=1
    )

def formulate_single_channel_regression_labels(lib, protein_of_interest):
    all_proteins = ALL_PROTEINS.copy()
    all_proteins.remove(protein_of_interest)
    lib["poi_log_fold"] = calculate_log_fold_across_proteins(
        lib, protein_of_interest, all_proteins
    )
    return lib["poi_log_fold"].to_numpy()


def formulate_binary_classification_labels(lib, protein_of_interest):
    all_proteins = ALL_PROTEINS.copy()
    all_proteins.remove(protein_of_interest)
    lib["poi_log_fold"] = calculate_log_fold_across_proteins(
        lib, protein_of_interest, all_proteins
    )
    return np.array(list(lib["poi_log_fold"].apply(lambda e: e > 0)))


def calculate_log_fold_across_proteins(
    lib,
    protein_of_interest,
    other_proteins,
):
    adder = partial(pd.Series.add, fill_value=0)

    def sum_over_replicates(protein_id, replicate_idxs=REPLICATE_IDXS):
        # Sums series over all three replicates of a protein specified
        return reduce(adder, [lib[protein_id + idx] for idx in replicate_idxs])

    # log fold change of protein of interest over all other proteins across replicates
    lib["poi_ratio"] = (sum_over_replicates(protein_of_interest) + 1) / (
        reduce(
            adder,
            [sum_over_replicates(protein_id) for protein_id in other_proteins],
        )
        + 1
    )
    return lib["poi_ratio"].apply(np.log)


def calculate_log_p_value(lib, protein_of_interest, other_protein):
    lib["pvalues"] = sp.stats.ttest_ind(
        lib[[protein_of_interest + idx for idx in REPLICATE_IDXS]].T,
        lib[[other_protein + idx for idx in REPLICATE_IDXS]].T,
    ).pvalue
    lib["log_pvalues"] = lib["pvalues"].apply(np.log)
    return lib["log_pvalues"]
