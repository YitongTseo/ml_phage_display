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

AA_FEATURE_DIM = 16


def shapley_analysis(model, X, investigate_X):
    shap.initjs()

    if len(investigate_X) > 50:
        print(
            "Heads up you just gave investigate_X w/ ",
            str(len(investigate_X)),
            " thats gonna take a long time",
        )

    def f(X_sample):
        return model.predict(decode_X(X_sample))

    def decode_X(X_sample):
        assert MAX_PEPTIDE_LEN == int(X_sample.shape[1] / AA_FEATURE_DIM)
        return X_sample.reshape(X_sample.shape[0], MAX_PEPTIDE_LEN, AA_FEATURE_DIM)

    def encode_X(X_sample):
        assert X_sample.shape[1] == MAX_PEPTIDE_LEN
        assert X_sample.shape[2] == AA_FEATURE_DIM
        return X_sample.reshape(X_sample.shape[0], MAX_PEPTIDE_LEN * AA_FEATURE_DIM)

    encoded_X = encode_X(X)
    encoded_investigate_X = encode_X(investigate_X)
    pdb.set_trace()
    # assert all(X == decode_X(encoded_X))

    # Here we select 50 samples for the "typical" feature values (used for calculating out features)
    # TODO(Yitong): Also see what it would be like to use kmeans function for calculating out features...
    np.random.seed(0)
    inds = np.random.choice(encoded_X.shape[0], 50, replace=False)
    summary = encoded_X[inds, :]

    # Then use 100 perterbation samples to estimate the SHAP values for a given prediction
    # Note that this requires 100 * 50 evaluations of the model.
    explainer = shap.KernelExplainer(f, summary)
    encoded_shap_values = explainer.shap_values(encoded_investigate_X, nsamples=100)
    shap_values = encode_X(encoded_shap_values)
    pdb.set_trace()

    if len(investigate_X) == 1:
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            features=encoded_investigate_X,
            feature_names=["a"] * 224,
            matplotlib=True,
            show=True,
        )
        pdb.set_trace()
    else:
        shap.summary_plot(shap_values, encoded_investigate_X)

    shap_values50 = explainer.shap_values(encoded_X[280:330, :], nsamples=500)
    shap.dependence_plot("alcohol", shap_values[0], encoded_X[:1])

    # p = shap.force_plot(explainer.expected_value, shap_values, X_display.iloc[299, :])

    shap_values = explainer.shap_values(X)
    i = 4
    # shap.partial_dependence_plot(
    #     "MedInc", model.predict, X100, ice=False,
    #     model_expected_value=True, feature_expected_value=True
    # )

    shap.summary_plot(shap_values, features=X, feature_names=X.columns)
    shap.summary_plot(shap_values, features=X, feature_names=X.columns, plot_type="bar")
    # # the waterfall_plot shows how we get from shap_values.base_values to model.predict(X)[sample_ind]
    shap.plots.waterfall(shap_values[0], max_display=14)
