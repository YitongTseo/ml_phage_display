import pathlib

HOME_DIRECTORY = pathlib.Path().absolute()

import sys
import os
import pdb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from utils.utils import cnt_c, find_3mer

from representation import seq_to_pro2vec, seq_to_RAA, seq_to_prop, MAX_PEPTIDE_LEN
from nn_models import train


def read_data(datafile="12ca5-MDM2-mCDH2-R3.csv", visualize=False):
    R3_lib = pd.read_csv(os.path.join(HOME_DIRECTORY, "data", datafile))
    R3_lib["Length"] = R3_lib.Peptide.apply(lambda x: len(x))
    R3_lib.columns

    R3_lib["is_dya"] = R3_lib["Peptide"].apply(lambda seq: find_3mer(seq, "DYA"))
    R3_lib["is_lle"] = R3_lib["Peptide"].apply(lambda seq: find_3mer(seq, "LLE"))
    R3_lib["c_cnt"] = R3_lib["Peptide"].apply(cnt_c)
    len(R3_lib.loc[R3_lib["c_cnt"] == 3])

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


def preprocessing(R3_lib, max_len=MAX_PEPTIDE_LEN):
    R3_lib.drop([1939, 4996], inplace=True)  # drop NaN sample

    R3_lib["Pro2Vec"] = R3_lib.Peptide.apply(seq_to_pro2vec)
    R3_lib["RAA"] = R3_lib.Peptide.apply(seq_to_RAA)
    R3_lib["prop"] = R3_lib.Peptide.apply(seq_to_prop)

    pro2vec = np.stack(R3_lib["Pro2Vec"].to_numpy())
    raa = np.stack(R3_lib["RAA"].to_numpy())
    prop = np.stack(R3_lib["prop"].to_numpy())

    # max length = 14, padding O
    pro2vec = pro2vec.reshape(-1, max_len, 1)
    raa = raa.reshape(-1, max_len, 1)

    X = np.concatenate((pro2vec, raa, prop), axis=-1)

    # Labels for neural networks
    R3_lib["12ca5_ratio"] = (
        R3_lib["12ca5 1"] + R3_lib["12ca5 2"] + R3_lib["12ca5 3"] + 1
    ) / (
        R3_lib["mCDH2 1"]
        + R3_lib["mCDH2 2"]
        + R3_lib["mCDH2 3"]
        + R3_lib["MDM2 1"]
        + R3_lib["MDM2 2"]
        + R3_lib["MDM2 3"]
        + 1
    )
    R3_lib["12ca5_log_ratio"] = R3_lib["12ca5_ratio"].apply(np.log)

    R3_lib["mCDH2_ratio"] = (
        R3_lib["mCDH2 1"] + R3_lib["mCDH2 2"] + R3_lib["mCDH2 3"] + 1
    ) / (
        R3_lib["12ca5 1"]
        + R3_lib["12ca5 2"]
        + R3_lib["12ca5 3"]
        + R3_lib["MDM2 1"]
        + R3_lib["MDM2 2"]
        + R3_lib["MDM2 3"]
        + 1
    )
    R3_lib["mCDH2_log_ratio"] = R3_lib["mCDH2_ratio"].apply(np.log)

    R3_lib["mdm2_ratio"] = (
        R3_lib["MDM2 1"] + R3_lib["MDM2 2"] + R3_lib["MDM2 3"] + 1
    ) / (
        R3_lib["12ca5 1"]
        + R3_lib["12ca5 2"]
        + R3_lib["12ca5 3"]
        + R3_lib["mCDH2 1"]
        + R3_lib["mCDH2 2"]
        + R3_lib["mCDH2 3"]
        + 1
    )
    R3_lib["mdm2_log_ratio"] = R3_lib["mdm2_ratio"].apply(np.log)

    # BIG TODO: Unhardcode MDM2 from the learning task
    y = R3_lib["mdm2_log_ratio"]
    # Cast as binary classification task
    y = np.array(list(y.apply(lambda e: e > 0)))
    X, y = shuffle(X, y, random_state=0)
    return X, y


R3_lib = read_data(datafile="12ca5-MDM2-mCDH2-R3.csv")
X, y = preprocessing(R3_lib)
result = train(X, y)
pdb.set_trace()
