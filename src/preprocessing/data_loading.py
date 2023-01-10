import pathlib
import sys

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
from preprocessing.y_representation import (
    formulate_binary_classification_labels,
    formulate_single_channel_regression_labels,
    formulate_two_channel_regression_labels,
)
from sklearn.preprocessing import StandardScaler
from enum import Enum
from preprocessing.X_representation import (
    seq_to_pro2vec,
    seq_to_RAA,
    seq_to_prop,
    MAX_PEPTIDE_LEN,
)
from itertools import compress
from preprocessing.X_representation_utils import AA_PROPERTY_ORDERING

FEATURE_LIST = [
    "Pro2Vec",
    "RAA",
] + AA_PROPERTY_ORDERING


class DATASET_TYPE(Enum):
    BINARY_CLASSIFICATION = 1
    LOG_FOLD_REGRESSION = 2
    JOINT_REGRESSION = 3


def read_data_and_preprocess(datafile="12ca5-MDM2-mCDH2-R3.csv"):
    lib = pd.read_csv(os.path.join(HOME_DIRECTORY, "data", datafile))

    # Filter "X" (i.e. unknown residue) containing peptides
    lib = lib[lib["Peptide"].str.contains("X") == False]

    # Add peptide metadata
    lib["Length"] = lib.Peptide.apply(lambda x: len(x))
    lib["is_dya"] = lib["Peptide"].apply(lambda seq: find_3mer(seq, "DYA"))
    lib["is_lle"] = lib["Peptide"].apply(lambda seq: find_3mer(seq, "LLE"))
    lib["c_cnt"] = lib["Peptide"].apply(cnt_c)

    # Precalculate peptide representations
    lib["Pro2Vec"] = lib.Peptide.apply(seq_to_pro2vec)
    lib["RAA"] = lib.Peptide.apply(seq_to_RAA)
    lib["prop"] = lib.Peptide.apply(seq_to_prop)
    return lib


def build_dataset(
    lib,
    protein_of_interest,
    other_protein,
    max_len=MAX_PEPTIDE_LEN,
    dataset_type=DATASET_TYPE.BINARY_CLASSIFICATION,
):
    pro2vec = np.stack(lib["Pro2Vec"].to_numpy())
    raa = np.stack(lib["RAA"].to_numpy())
    prop = np.stack(lib["prop"].to_numpy())
    peptides = lib["Peptide"].to_list()

    # max length = 14, padding O
    pro2vec = pro2vec.reshape(-1, max_len, 1)
    raa = raa.reshape(-1, max_len, 1)
    assert FEATURE_LIST.index("Pro2Vec") == 0 and FEATURE_LIST.index("RAA") == 1
    # Concat Pro2Vec as the first element, RAA as the second element, and properties as elements 3-16
    X = np.concatenate((pro2vec, raa, prop), axis=-1)
    # TODO(Yitong): Do we want to normalize X?

    if dataset_type == DATASET_TYPE.BINARY_CLASSIFICATION:
        y = formulate_binary_classification_labels(lib, protein_of_interest)
    elif dataset_type in (
        # TODO(Yitong): We dont need both of these when they do the same thing...
        DATASET_TYPE.LOG_FOLD_REGRESSION,
        DATASET_TYPE.JOINT_REGRESSION,
    ):
        y = formulate_two_channel_regression_labels(
            lib, protein_of_interest, other_protein
        )
        # Remove nans ... aka when p-value = 0
        nans_mask = ~np.isnan(y).any(axis=1)
        # Remove negative infinity ... aka when p-value = 1
        neg_inf_mask = ~(y == -np.inf).any(axis=1)
        mask = np.logical_and(nans_mask, neg_inf_mask)
        X = X[mask]
        peptides = list(compress(peptides, mask))
        y = y[mask]

        # log p-value into -log p-value
        y[:, 0] = -y[:, 0]

        # Normalize
        scaler = StandardScaler()
        scaler.fit(y)
        y = scaler.transform(y)
        log_P_5percent, log_FC_zero = scaler.transform([[-np.log10(0.05), 0]])[0]
        print(
            " - log P value cutoff is {}, and log FC value cutoff is {}".format(
                log_P_5percent, log_FC_zero
            )
        )

    assert (
        len(np.argwhere(np.isnan(y))) == 0
    ), "we dont want any nans in our y-label dataset"
    X, y, peptides = shuffle(X, y, peptides, random_state=0)
    return X, y, peptides
