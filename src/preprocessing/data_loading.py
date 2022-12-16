import pathlib
import sys
import pdb

HOME_DIRECTORY = pathlib.Path().absolute()
sys.path.append(str(HOME_DIRECTORY) + "/src")

import sys
import os
import pdb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from utils.utils import cnt_c, find_3mer
from functools import reduce, partial

from preprocessing.representation import (
    seq_to_pro2vec,
    seq_to_RAA,
    seq_to_prop,
    MAX_PEPTIDE_LEN,
)


def read_data(datafile="12ca5-MDM2-mCDH2-R3.csv", visualize=False):
    R3_lib = pd.read_csv(os.path.join(HOME_DIRECTORY, "data", datafile))
    R3_lib["Length"] = R3_lib.Peptide.apply(lambda x: len(x))
    # R3_lib.columns

    R3_lib["is_dya"] = R3_lib["Peptide"].apply(lambda seq: find_3mer(seq, "DYA"))
    R3_lib["is_lle"] = R3_lib["Peptide"].apply(lambda seq: find_3mer(seq, "LLE"))
    R3_lib["c_cnt"] = R3_lib["Peptide"].apply(cnt_c)
    # len(R3_lib.loc[R3_lib["c_cnt"] == 3])

    if visualize:
        sns.histplot(R3_lib["c_cnt"])
        plt.title("Cysteine counts")
        plt.show()

        sns.histplot(R3_lib.loc[R3_lib.is_dya == True]["12ca5_log_ratio"], binwidth=0.2)
        plt.title("DYA peptide")
        plt.show()

        sns.histplot(
            R3_lib.loc[R3_lib.is_lle == True]["12ca5_log_ratio"],
            binwidth=0.2,
            alpha=0.5,
        )
        plt.title("LLE peptide")
        plt.show()

        print((R3_lib.loc[R3_lib.is_lle == True]["12ca5_log_ratio"] > 0).sum())
        print((R3_lib.loc[R3_lib.is_lle == True]["12ca5_log_ratio"] <= 0).sum())

        sns.histplot(R3_lib["12ca5_log_ratio"], binwidth=0.2, alpha=0.7)
        plt.title("all peptide")
        plt.show()
        # sns.histplot(R3_lib.loc[R3_lib.is_dya==True]['12ca5_log_ratio'], binwidth=0.2)
    return R3_lib


def preprocessing(
    R3_lib,
):
    # TODO: we should just remove these two rows with NaN samples from dataset
    R3_lib.drop([1939, 4996], inplace=True)  # drop NaN sample
    R3_lib["Pro2Vec"] = R3_lib.Peptide.apply(seq_to_pro2vec)
    R3_lib["RAA"] = R3_lib.Peptide.apply(seq_to_RAA)
    R3_lib["prop"] = R3_lib.Peptide.apply(seq_to_prop)
    return R3_lib


def build_dataset(
    R3_lib,
    protein_of_interest,
    # TODO: Remove Cadherin since it's not in scope of bioarxiv submission
    all_proteins=["12ca5", "MDM2", "mCDH2",],  
    max_len=MAX_PEPTIDE_LEN,
):
    pro2vec = np.stack(R3_lib["Pro2Vec"].to_numpy())
    raa = np.stack(R3_lib["RAA"].to_numpy())
    prop = np.stack(R3_lib["prop"].to_numpy())

    # max length = 14, padding O
    pro2vec = pro2vec.reshape(-1, max_len, 1)
    raa = raa.reshape(-1, max_len, 1)
    X = np.concatenate((pro2vec, raa, prop), axis=-1)

    def sum_over_replicates(protein_id, replicate_idxs=(" 1", " 2", " 3")):
        # Sums series over all three replicates of a protein specified
        adder = partial(pd.Series.add, fill_value=0)
        return reduce(
            adder, [R3_lib[protein_id + idx] for idx in replicate_idxs]
        )

    other_proteins_not_of_interest = all_proteins
    other_proteins_not_of_interest.remove(protein_of_interest)
    # Labels for neural networks
    R3_lib["poi_ratio"] = (
        sum_over_replicates(protein_of_interest) + 1
    ) / (
        sum_over_replicates(other_proteins_not_of_interest[0]) +
        sum_over_replicates(other_proteins_not_of_interest[1]) + 
        1
    )
    R3_lib["poi_log_ratio"] = R3_lib["poi_ratio"].apply(np.log)
    y = R3_lib["poi_log_ratio"]

    # Cast as binary classification task
    y = np.array(list(y.apply(lambda e: e > 0)))
    X, y = shuffle(X, y, random_state=0)
    return X, y
