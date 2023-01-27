import numpy as np
from functools import partial
import matplotlib.pyplot as plt

import tqdm
from dataclasses import asdict
import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import preprocessing.data_loading as data_loading
import models.experiment as experiment
import models.rnn as rnn
import analysis.umap_analysis as umap
import analysis.scatter_plots as scatter_plots
import analysis.evaluation as evaluation
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
import copy


"""
These model calls shouldn't be housed here...
"""


def train_random_forest(X_train, y_train, X_test):
    processed_X_train = X_train.reshape(X_train.shape[0], -1)
    processed_X_test = X_test.reshape(X_test.shape[0], -1)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(processed_X_train, y_train)
    return rf.predict(processed_X_test)


def train_bilstm_model(X_train, y_train, X_test):
    model = experiment.BinaryClassificationExperiment().train(
        X_train,
        y_train,
        rnn.Joint_BinaryClassificationRNN_gelu,
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
    )
    y_pred = model(X_test) > 0.5
    return y_pred


def negative_predictions(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn + fn


def positive_predictions(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tp + tp


def model_comparison_matrix_visualization(
    protein_of_interest="MDM2",
    other_protein="12ca5",
    test_train_split=0.2,
    plot_titles_to_metrics={
        # "Precision": metrics.precision_score,
        # "Recall": metrics.recall_score,
        "Accuracy": metrics.accuracy_score,
        # "Total Positive Predictions\n(true pos + false pos)": positive_predictions,
        # "Total Negative Predictions\n(true neg + false neg)": negative_predictions,
    },
    models=[
        {
            "title": "Random\nForest",
            "model_call": train_random_forest,
        },
        {
            "title": "BiLSTM\nNeural\nNetwork",
            "model_call": train_bilstm_model,
        },
    ],
    representations=[
        {
            "representation": [
                data_loading.AA_REPRESENTATION.PRO2VEC,
            ],
            "title": "Pro2Vec (1 dim)",
        },
        {
            "representation": [
                data_loading.AA_REPRESENTATION.RAA,
            ],
            "title": "RAA (1 dim)",
        },
        {
            "representation": [
                data_loading.AA_REPRESENTATION.PHYSIOCHEM_PROPERTIES,
            ],
            "title": "Physiochem Properties (14 dim)",
        },
        {
            "representation": [
                data_loading.AA_REPRESENTATION.ONE_HOT,
            ],
            "title": "One Hot (21 dim)",
        },
        {
            "representation": [
                data_loading.AA_REPRESENTATION.PRO2VEC,
                data_loading.AA_REPRESENTATION.RAA,
                data_loading.AA_REPRESENTATION.PHYSIOCHEM_PROPERTIES,
                data_loading.AA_REPRESENTATION.ONE_HOT,
            ],
            "title": "Pro2Vec + RAA +\nPhysiochemical Properties +\nOne Hot (37 dim)",
        },
    ],
):
    R3_lib = data_loading.read_data_and_preprocess(datafile="12ca5-MDM2-mCDH2-R3.csv")
    fc_results = {
        plot_title: {
            representation_params["title"]: list()
            for representation_params in representations
        }
        for plot_title in plot_titles_to_metrics.keys()
    }
    pval_results = copy.deepcopy(fc_results)

    for representation_params in tqdm.tqdm(representations):
        for model_params in tqdm.tqdm(models):
            X, y, _, _, _, _, _, = data_loading.build_dataset(
                R3_lib,
                protein_of_interest=protein_of_interest,
                other_protein=other_protein,
                aa_representations=representation_params["representation"],
            )
            (X_train, X_test, y_train, y_test) = train_test_split(
                X,
                y,
                test_size=test_train_split,
                shuffle=True,
                random_state=5,
            )
            y_pred = model_params["model_call"](X_train, y_train, X_test)

            for plot_title, metric in plot_titles_to_metrics.items():
                fc_results[plot_title][representation_params["title"]].append(
                    metric(y_test[:, 1], y_pred[:, 1])
                )
                pval_results[plot_title][representation_params["title"]].append(
                    metric(y_test[:, 0], y_pred[:, 0])
                )

    def visualize(plot_title, specific_result):
        matrix = np.array(
            [np.array(specific_result[p["title"]]) for p in representations]
        )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(matrix, interpolation="nearest")
        fig.colorbar(cax)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([p["title"] for p in models])
        ax.set_yticks(range(len(representations)))
        ax.set_yticklabels([p["title"] for p in representations])

        # Loop over data dimensions and create text annotations.
        for row_idx, row in enumerate(matrix):
            for col_idx in range(len(row)):
                ax.text(
                    col_idx,
                    row_idx,
                    np.round(matrix[row_idx, col_idx], decimals=3),
                    ha="center",
                    va="center",
                    color="w",
                )
        ax.set_ylabel("Representation")
        ax.set_xlabel("Model")
        ax.set_title(plot_title)
        plt.tight_layout(pad=1)
        plt.show()

    for plot_title in plot_titles_to_metrics.keys():
        visualize(
            f"{protein_of_interest} vs. {other_protein}: P-value {plot_title}",
            pval_results[plot_title],
        )
        visualize(
            f"{protein_of_interest} vs. {other_protein}: Fold Change {plot_title}",
            fc_results[plot_title],
        )
