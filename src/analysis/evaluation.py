import pathlib
import sys

HOME_DIRECTORY = pathlib.Path().absolute()
sys.path.append(str(HOME_DIRECTORY) + "/src")
import sklearn.metrics as metrics

from sklearn.metrics import confusion_matrix

import models.experiment as experiment
import models.rnn as rnn
import numpy as np
from tensorflow import keras
from functools import partial
from analysis.scatter_plots import plot_relations_in_3D, plot_fancy_hexbin_relations
from utils.utils import seed_everything


def pval_filter_ranking_lambda(x):
    return x[1] + x[2]
    # if x[0] > - np.log(0.01): # else try 2 #- np.log(0.01):
    #     return x[1] + x[2]
    # else:
    #     return 0


def cross_validate_and_benchmark(
    X,
    y_raw,
    peptides,
    model_save_name,
    loss,
    motif_dectection_func,
    load_trained_model=False,
    calculate_proxy_uncertainty=True,
    ranking_lambda=pval_filter_ranking_lambda,
    plot_x_idx=1,
    plot_y_idx=0,
    width=64,
):
    seed_everything(0)
    results = experiment.Experiment().run_cross_validation_experiment(
        X=X,
        y=y_raw,
        model_architecture=partial(
            rnn.ThreeChannelRegressionRNN_gelu, loss=loss, width=width
        ),
        optimizer=partial(
            keras.optimizers.Adam, learning_rate=0.0005, weight_decay=0.5
        ),
        n_splits=10,
        load_trained_model=load_trained_model,
        model_save_name=model_save_name,
        normalize=True,
        batch_size=128,
        num_epochs=5,
    )

    y_pred = np.vstack(result.y_pred_rescaled for result in results)
    y_true = np.vstack(result.y_test for result in results)
    # Check that the cross folds experiment returns our y_true in the same order as before
    assert (y_true == y_raw).all()

    all_positives = np.array(
        [1.0 if motif_dectection_func(pep) else 0.0 for pep in peptides]
    )
    if calculate_proxy_uncertainty:
        # Calculate with dropout on...
        pred_100_fold = []
        for result in results:
            pred_100_fold.append(
                np.array(
                    [
                        result.trained_model(result.X_test, training=True)
                        for _ in range(100)
                    ]
                )
            )
        pred_100_fold = np.concatenate(pred_100_fold, axis=1)

        mean = np.mean(pred_100_fold, axis=0)
        variance = np.std(pred_100_fold, axis=0)
        uncertainty = np.mean(variance, axis=1)

        mdm2_ordering = [ranking_lambda(pred) for pred in mean]
        plot_relations_in_3D(
            plot_x_idx,
            plot_y_idx,
            datapoints=mean,
            title="Predicted Hits Coloring",
            ordering=mdm2_ordering,
            all_positives=all_positives,
            uncertainty=uncertainty,
        )
        y_pred = mean
    else:
        mdm2_ordering = [ranking_lambda(pred) for pred in y_pred]
        # plot_relations(
        #     plot_x_idx,
        #     plot_y_idx,
        #     datapoints=y_pred,
        #     ordering=mdm2_ordering,
        #     all_positives=all_positives,
        #     kind="scatter",
        # )
    plot_fancy_hexbin_relations(
        plot_x_idx,
        plot_y_idx,
        datapoints=y_pred,
        ordering=mdm2_ordering,
        all_positives=None,  # all_positives,
        line_color="#F94040",
        vals=[
            "Predicted -log(P-value)",
            "Predicted log(Fold Change)",
            "Predicted Enrichment Ratio",
        ],
    )
    plot_fancy_hexbin_relations(
        plot_x_idx,
        plot_y_idx,
        datapoints=y_pred,
        ordering=None,  # mdm2_ordering,
        all_positives=all_positives,
        line_color="#F94040",
        vals=[
            "Predicted -log(P-value)",
            "Predicted log(Fold Change)",
            "Predicted Enrichment Ratio",
        ],
    )
    return mdm2_ordering, y_pred


def classification_evaluation_joint(y_true, y_pred):
    # y_true & y_test shape (# samples, 2) for both p-value & fc
    print("P-VAL")
    tn, fp, fn, tp = confusion_matrix(y_true[:, 0], y_pred[:, 0]).ravel()
    print("\t accuracy: ", metrics.accuracy_score(y_true[:, 0], y_pred[:, 0]))
    print("\t precision: ", metrics.precision_score(y_true[:, 0], y_pred[:, 0]))
    print("\t recall: ", metrics.recall_score(y_true[:, 0], y_pred[:, 0]))
    print("\t # neg predictions: ", tn + fn)
    print("\t # pos predictions: ", tp + fp)

    print("Log Fold")
    tn, fp, fn, tp = confusion_matrix(y_true[:, 1], y_pred[:, 1]).ravel()
    print("\t accuracy: ", metrics.accuracy_score(y_true[:, 1], y_pred[:, 1]))
    print("\t precision: ", metrics.precision_score(y_true[:, 1], y_pred[:, 1]))
    print("\t recall: ", metrics.recall_score(y_true[:, 1], y_pred[:, 1]))
    print("\t # neg predictions: ", tn + fn)
    print("\t # pos predictions: ", tp + fp)


def classifcation_evaluation(y_train, y_pred, y_cutoff):
    y_train = y_train > y_cutoff
    y_pred = y_pred > 0.5
    tn, fp, fn, tp = confusion_matrix(y_train > 0, y_pred).ravel()
    acc, pre, rec = [], [], []
    print(tn, fp, fn, tp)
    print("accuracy", (tn + tp) / (tn + fp + fn + tp))
    acc.append((tn + tp) / (tn + fp + fn + tp))
    print("precision", tp / (tp + fp))
    pre.append(tp / (tp + fp))
    print("recall", tp / (tp + fn))
    rec.append(tp / (tp + fn))


def single_regression_evaluation(y_train, y_pred, y_cutoff):
    y_train = y_train > y_cutoff
    y_pred = y_pred > 0.5
    tn, fp, fn, tp = confusion_matrix(y_train > 0, y_pred).ravel()
    acc, pre, rec = [], [], []
    print(tn, fp, fn, tp)
    print("accuracy", (tn + tp) / (tn + fp + fn + tp))
    acc.append((tn + tp) / (tn + fp + fn + tp))
    print("precision", tp / (tp + fp))
    pre.append(tp / (tp + fp))
    print("recall", tp / (tp + fn))
    rec.append(tp / (tp + fn))


def joint_regression_evaluation(
    y1_train, y1_pred, y1_cutoff, y2_train, y2_pred, y2_cutoff
):
    y1_train = y1_train > y1_cutoff
    y2_train = y2_train > y2_cutoff
    y1_pred = y1_pred > 0.5
    y2_pred = y2_pred > 0.5
    tn, fp, fn, tp = confusion_matrix(
        (y1_train) * (y2_train), (y1_pred) * (y2_pred)
    ).ravel()
    acc, pre, rec = [], [], []
    print(tn, fp, fn, tp)
    print("accuracy", (tn + tp) / (tn + fp + fn + tp))
    acc.append((tn + tp) / (tn + fp + fn + tp))
    print("precision", tp / (tp + fp))
    pre.append(tp / (tp + fp))
    print("recall", tp / (tp + fn))
    rec.append(tp / (tp + fn))
