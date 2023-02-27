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
    seq_to_onehot,
)
from itertools import compress
from preprocessing.X_representation_utils import (
    AA_PROPERTY_ORDERING,
    AA_ONE_HOT_ORDERING,
)
from typing import List

FEATURE_LIST = []
LOG_FC_CUTOFF = 0
LOG_PVALUE_CUTOFF = -np.log10(0.05)
MAX_PEPTIDE_LEN = 14


class AA_REPRESENTATION(Enum):
    PRO2VEC = 1
    RAA = 2
    PHYSIOCHEM_PROPERTIES = 3
    ONE_HOT = 4


def read_data_and_preprocess(datafile, nmer_filter=None):
    lib = pd.read_csv(os.path.join(HOME_DIRECTORY, "data", datafile))

    # Filter "X" (i.e. unknown residue) containing peptides
    lib = lib[lib["Peptide"].str.contains("X") == False]

    # Add peptide metadata
    lib["Length"] = lib.Peptide.apply(lambda x: len(x))
    lib["is_dya"] = lib["Peptide"].apply(lambda seq: find_3mer(seq, "DYA"))
    lib["is_lle"] = lib["Peptide"].apply(lambda seq: find_3mer(seq, "LLE"))
    lib["c_cnt"] = lib["Peptide"].apply(cnt_c)

    if nmer_filter:
        lib = lib[lib["Length"] == nmer_filter]
    # Precalculate peptide representations
    lib["Pro2Vec"] = lib.Peptide.apply(seq_to_pro2vec)
    lib["RAA"] = lib.Peptide.apply(seq_to_RAA)
    lib["prop"] = lib.Peptide.apply(seq_to_prop)
    lib["onehot"] = lib.Peptide.apply(seq_to_onehot)
    return lib


def build_dataset(
    lib,
    protein_of_interest,
    other_protein,
    aa_representations: List[AA_REPRESENTATION] = [
        AA_REPRESENTATION.PRO2VEC,
        AA_REPRESENTATION.RAA,
        AA_REPRESENTATION.PHYSIOCHEM_PROPERTIES,
        AA_REPRESENTATION.ONE_HOT,
    ],
):
    pro2vec = np.stack(lib["Pro2Vec"].to_numpy())
    raa = np.stack(lib["RAA"].to_numpy())
    prop = np.stack(lib["prop"].to_numpy())
    onehot = np.stack(lib["onehot"].to_numpy())
    peptides = lib["Peptide"].to_list()

    # max length = 14, padding O
    pro2vec = pro2vec.reshape(-1, MAX_PEPTIDE_LEN, 1)
    raa = raa.reshape(-1, MAX_PEPTIDE_LEN, 1)

    # By creating feature_list on the fly we can ensure it is the correct ordering
    FEATURE_LIST = []
    X = np.array([])
    X.shape = (len(lib), MAX_PEPTIDE_LEN, 0)
    if AA_REPRESENTATION.PRO2VEC in aa_representations:
        X = np.concatenate((X, pro2vec), axis=-1)
        FEATURE_LIST.append("Pro2Vec")
    if AA_REPRESENTATION.RAA in aa_representations:
        X = np.concatenate((X, raa), axis=-1)
        FEATURE_LIST.append("RAA")
    if AA_REPRESENTATION.PHYSIOCHEM_PROPERTIES in aa_representations:
        X = np.concatenate((X, prop), axis=-1)
        FEATURE_LIST += AA_PROPERTY_ORDERING
    if AA_REPRESENTATION.ONE_HOT in aa_representations:
        X = np.concatenate((X, onehot), axis=-1)
        FEATURE_LIST += AA_ONE_HOT_ORDERING

    y_raw = formulate_two_channel_regression_labels(
        lib, protein_of_interest, other_protein
    )
    # Remove nans ... aka when p-value = 0
    nans_mask = ~np.isnan(y_raw).any(axis=1)
    # Remove negative infinity ... aka when p-value = 1
    neg_inf_mask = ~(y_raw == -np.inf).any(axis=1)
    mask = np.logical_and(nans_mask, neg_inf_mask)
    X = X[mask]
    peptides = list(compress(peptides, mask))
    y_raw = y_raw[mask]

    # log p-value into -log p-value
    y_raw[:, 0] = -y_raw[:, 0]

    # Normalize
    scaler = StandardScaler()
    scaler.fit(y_raw)
    y_raw = scaler.transform(y_raw)
    y_classes = np.copy(y_raw)

    log_P_5percent, log_FC_zero = scaler.transform(
        [[LOG_PVALUE_CUTOFF, LOG_FC_CUTOFF]]
    )[0]
    print(
        " - log P value cutoff is {}, and log FC value cutoff is {}".format(
            log_P_5percent, log_FC_zero
        )
    )

    # Create y_classes based off of thresholds determined after scaling
    y_classes[:, 1] = y_classes[:, 1] > log_FC_zero
    y_classes[:, 0] = y_classes[:, 0] > log_P_5percent

    assert (
        len(np.argwhere(np.isnan(y_raw))) == 0
    ), "we dont want any nans in our y-label dataset"
    X, y_classes, y_raw, peptides = shuffle(
        X, y_classes, y_raw, peptides, random_state=0
    )

    # create the other peptide data by reversing fold change values
    # this is acceptable because the class threshold (log_FC_zero)
    # is set to 0, hence classes are symmetric
    other_y_classes = np.copy(y_classes)
    other_y_classes[:, 1] = other_y_classes[:, 1] == 0

    other_y_raw = np.copy(y_raw)
    other_y_raw[:, 1] = -other_y_raw[:, 1]
    return X, y_classes, y_raw, other_y_classes, other_y_raw, peptides, FEATURE_LIST
