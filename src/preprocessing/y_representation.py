from functools import reduce, partial
import pandas as pd
import numpy as np
import scipy as sp
import os
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


def calculate_er_values(
    protein_of_interest,
    round_datafiles=[
        "data/12ca5-MDM2-R1.csv",
        "data/12ca5-MDM2-R2.csv",
        "data/12ca5-MDM2-R3.csv",
    ],
    save_output=True,
):
    """
    Refer to the equation defined in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9872955/
    """
    round_dfs = [pd.read_csv(round_datafile) for round_datafile in round_datafiles]

    cols = [f"{protein_of_interest}{replicate_idx}" for replicate_idx in REPLICATE_IDXS]

    # TODO: this will break with replicates > 3
    for round_idx, df in enumerate(round_dfs):
        df[f"{protein_of_interest} sum"] = df[cols].sum(axis=1)
        df = df[["Peptide", f"{protein_of_interest} sum"]]
        round_dfs[round_idx] = df

    # Merge and drop duplicates for all DataFrames in (reversed!) round_dfs
    # TODO: this will break for any protein with > 3 rounds...
    df_merged = reduce(
        lambda left, right: pd.merge(left, right, on="Peptide", how="left"),
        round_dfs[::-1],
    )
    df_merged = df_merged.drop_duplicates(subset=["Peptide"])
    df_merged = df_merged.reset_index(drop=True)

    # Make sure every cell is not NaN, and add a 1 to every cell.
    df_merged = df_merged.fillna(0)
    numerical_cols = df_merged.select_dtypes(include="number").columns
    df_merged[numerical_cols] = df_merged[numerical_cols].add(1.0)

    # TODO: this will break for any protein with > 3 rounds...
    df_merged[f"{protein_of_interest} sum_x"] = df_merged[
        f"{protein_of_interest} sum_x"
    ] / (df_merged[f"{protein_of_interest} sum_x"].sum())
    df_merged[f"{protein_of_interest} sum_y"] = df_merged[
        f"{protein_of_interest} sum_y"
    ] / (df_merged[f"{protein_of_interest} sum_y"].sum())
    df_merged[f"{protein_of_interest} sum"] = df_merged[
        f"{protein_of_interest} sum"
    ] / (df_merged[f"{protein_of_interest} sum"].sum())

    df_merged["ER"] = (
        np.log2(
            (
                df_merged[f"{protein_of_interest} sum_x"]
                / (df_merged[f"{protein_of_interest} sum_y"])
            )
        )
    ) + (
        np.log2(
            (df_merged[f"{protein_of_interest} sum_y"])
            / (df_merged[f"{protein_of_interest} sum"])
        )
    )
    if save_output:
        df_merged.to_csv(f"data/{protein_of_interest}_merged_ER.csv")
    return df_merged

# If the ER pre-calculated files don't exist yet then generate them
if not (os.path.exists("data/12ca5_merged_ER.csv")) or not (
    os.path.exists(f"data/MDM2_merged_ER")
):
    print('Generating ER datafiles!')
    calculate_er_values(protein_of_interest="12ca5")
    calculate_er_values(protein_of_interest="MDM2")
