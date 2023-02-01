# TODO(Yitong): Remove these comments eventually
# For an introduction to Shapley additives I read these blogs:
# https://towardsdatascience.com/a-novel-approach-to-feature-importance-shapley-additive-explanations-d18af30fc21b#5744
# https://towardsdatascience.com/explain-any-models-with-the-shap-values-use-the-kernelexplainer-79de9464897a
# https://towardsdatascience.com/shapling-around-a-custom-network-636b97b40628

# Also referring to the documentation
# https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html
# https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/neural_networks/Census%20income%20classification%20with%20Keras.html
# https://shap-lrjball.readthedocs.io/en/latest/generated/shap.DeepExplainer.html
# https://shap-lrjball.readthedocs.io/en/latest/example_notebooks/kernel_explainer/ImageNet%20VGG16%20Model%20with%20Keras.html
import pandas as pd
import numpy as np
import shap
import pdb
from preprocessing.X_representation import MAX_PEPTIDE_LEN
from preprocessing.data_loading import FEATURE_LIST
from enum import Enum
import matplotlib.pyplot as plt
import sklearn
from itertools import product


class INVESTIGATION_TYPE(Enum):
    BY_AMINO_ACID = 1
    BY_FEATURE = 2
    BY_AMINO_ACID_AND_FEATURE = 3


AA_FEATURE_DIM = len(FEATURE_LIST)

def auto_cohorts(shap_values, max_cohorts):
    """
    This uses a DecisionTreeRegressor to build a group of cohorts with similar SHAP values.
    Implementation lifted from https://github.com/slundberg/shap/blob/master/shap/_explanation.py
    For the sake of opening up the hood and retrieving the instance names per cohort
    """

    # fit a decision tree that well spearates the SHAP values
    m = sklearn.tree.DecisionTreeRegressor(max_leaf_nodes=max_cohorts)
    m.fit(shap_values.data, shap_values.values)

    # group instances by their decision paths
    paths = m.decision_path(shap_values.data).toarray()
    unique_paths = np.unique(m.decision_path(shap_values.data).todense(), axis=0)
    path_names = []

    # mark each instance with a path name
    for i in range(shap_values.shape[0]):
        name = ""
        for j in range(len(paths[i])):
            if paths[i, j] > 0:
                feature = m.tree_.feature[j]
                threshold = m.tree_.threshold[j]
                val = shap_values.data[i, feature]
                if feature >= 0:
                    name += str(shap_values.feature_names[feature])
                    if val < threshold:
                        name += " < "
                    else:
                        name += " >= "
                    name += str(threshold) + " & "
        path_names.append(name[:-3])  # the -3 strips off the last unneeded ' & '
    path_names = np.array(path_names)

    # split the instances into cohorts by their path names
    cohorts = {}
    for name in np.unique(path_names):
        cohorts[name] = (
            shap_values[path_names == name],
            # Include instance names so we know which peptides are in which cohort
            [
                shap_values.instance_names[idx]
                for idx in np.where(path_names == name)[0]
            ],
        )
    return cohorts


def shapley_analysis(
    model,
    X,
    investigation_target,
    investigation_type=INVESTIGATION_TYPE.BY_FEATURE,
    num_background_samples=100,
    num_perturbation_samples=1000,
    aa_feature_dim=AA_FEATURE_DIM,
    peptide_dim=MAX_PEPTIDE_LEN,
    show_plot=True,
):
    shap.initjs()
    investigate_X = np.array([x[0] for x in investigation_target])
    investigate_peptides = [x[1] for x in investigation_target]
    if len(investigate_X) > 50:
        print(
            "Heads up you just gave investigate_X w/ ",
            str(len(investigate_X)),
            " thats gonna take a long time",
        )

    def f(X_sample):
        pred = model.predict(decode_X(X_sample))
        # Multiply predicted FC confidence & predicted P-Value confidence
        return pred[:,0] * pred[:, 1]

    def decode_X(X_sample):
        assert peptide_dim == int(X_sample.shape[1] / aa_feature_dim)
        return X_sample.reshape(X_sample.shape[0], peptide_dim, aa_feature_dim)

    def encode_X(X_sample):
        assert X_sample.shape[1] == peptide_dim
        assert X_sample.shape[2] == aa_feature_dim
        return X_sample.reshape(X_sample.shape[0], peptide_dim * aa_feature_dim)

    encoded_X = encode_X(X)
    encoded_investigate_X = encode_X(investigate_X)
    assert (X == decode_X(encoded_X)).all()

    # Here we select "num_background_samples" samples for the "typical" feature values (used for calculating out features)
    # TODO(Yitong): Also see what it would be like to use kmeans function for calculating out features...
    np.random.seed(0)
    inds = np.random.choice(encoded_X.shape[0], num_background_samples, replace=False)
    summary = encoded_X[inds, :]

    # Then use "num_perturbation_samples" perterbation samples to estimate the SHAP values for a given prediction
    # Note that this requires (num_background_samples * num_perturbation_samples) evaluations of the model.
    explainer = shap.KernelExplainer(f, summary)
    encoded_shap_values = explainer.shap_values(
        encoded_investigate_X, nsamples=num_perturbation_samples, silent= True
    )
    shap_values = decode_X(encoded_shap_values)

    if len(encoded_investigate_X) == 1:
        print(
            "Expected value: ",
            explainer.expected_value,
            "shapely value sum ",
            shap_values.sum(),
        )
        pred = model(investigate_X)
        print("Model eval: ", pred[:,0] * pred[:, 1])
        print(
            "Does this match the sum of shapely values + expected value (\Sum(phi_i) + phi_0)? ",
            shap_values.sum() + explainer.expected_value,
        )
        if investigation_type == INVESTIGATION_TYPE.BY_FEATURE:
            attribution = shap_values.sum(axis=1)[0]
            if show_plot:
                shap.force_plot(
                    explainer.expected_value,
                    attribution,
                    features=investigate_X.mean(axis=1)[0],
                    feature_names=FEATURE_LIST,
                    matplotlib=True,
                    show=True,
                )
        elif investigation_type == INVESTIGATION_TYPE.BY_AMINO_ACID:
            peptide_aas = list(investigate_peptides[0])
            # Only show attribution over the peptide length
            attribution = shap_values.sum(axis=2)[0][: len(peptide_aas)]
            if show_plot:
                plt.bar(
                    range(len(peptide_aas)),
                    attribution,
                    color="maroon",
                    width=0.4,
                )
                plt.xlabel("Amino Acid")
                plt.ylabel("Shap Value")
                plt.title(
                    "Attribution by amino acid for "
                    + investigate_peptides[0]
                    + " \n(negative value tend toward 12ca5 binders, positive values tend toward MDM2 binders)"
                )
                plt.xticks(np.asarray([i for i in range(len(peptide_aas))]), peptide_aas)
                plt.show()
        return attribution
    else:
        if investigation_type == INVESTIGATION_TYPE.BY_FEATURE:
            max_display = aa_feature_dim
            shap_values = shap_values.sum(axis=1)
            features = investigate_X.mean(axis=1)
            feature_names = FEATURE_LIST
        elif investigation_type == INVESTIGATION_TYPE.BY_AMINO_ACID:
            max_display = peptide_dim
            shap_values = shap_values.sum(axis=2)
            features = investigate_X.mean(axis=2)
            feature_names = ["aa_" + str(idx) for idx in range(peptide_dim)]
        elif investigation_type == INVESTIGATION_TYPE.BY_AMINO_ACID_AND_FEATURE:
            max_display = aa_feature_dim * peptide_dim
            shap_values = encoded_shap_values
            features = encoded_investigate_X
            feature_names = [
                tup[0] + ":" + tup[1]
                for tup in list(
                    product(
                        ["aa_" + str(idx) for idx in range(peptide_dim)],
                        FEATURE_LIST,
                    )
                )
            ]
        explanation = shap.Explanation(
            shap_values,
            data=features,
            feature_names=feature_names,
            instance_names=investigate_peptides,
        )
        shap.summary_plot(explanation, features, cmap="plasma")

        # Let's do some cohorting!
        def cohort_analysis(max_cohorts):
            shap.plots.bar(explanation.cohorts(max_cohorts).abs.mean(0))
            print(
                "\t" + str(max_cohorts) + " Cohort split: ",
                {k: v[1] for (k, v) in auto_cohorts(explanation, max_cohorts).items()},
            )

        cohort_analysis(2)
        cohort_analysis(3)
        cohort_analysis(4)
        cohort_analysis(5)
        cohort_analysis(6)
