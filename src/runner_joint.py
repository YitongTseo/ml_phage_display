import pdb
from preprocessing.data_loading import (
    read_data_and_preprocess,
    build_dataset,
    DATASET_TYPE,
)
from models.experiment import (
    TwoChannelRegressionExperiment,
    BinaryClassificationExperiment,
    SingleChannelRegressionExperiment,
)
from models.rnn import (
    TwoChannelRegressionRNN,
    SingleChannelRegressionRNN,
    BinaryClassificationRNN,
)
from tensorflow import keras
import matplotlib.pyplot as plt
import preprocessing.y_representation as y_representation
from analysis.shapley_additive_analysis import shapley_analysis, INVESTIGATION_TYPE
import numpy as np
from analysis.scatter_plots import show_volcano


R3_lib = read_data_and_preprocess(datafile="12ca5-MDM2-mCDH2-R3.csv")
protein_of_interest = "12ca5"
other_protein = "MDM2"
X, y, peptides = build_dataset(
    R3_lib,
    protein_of_interest=protein_of_interest,
    other_protein=other_protein,
    dataset_type=DATASET_TYPE.JOINT_REGRESSION,
)

result = TwoChannelRegressionExperiment().run_adhoc_experiment(
    X, y, TwoChannelRegressionRNN, load_trained_model=True
)
show_volcano(
    y, protein_of_interest, other_protein, title_addendum="True "
)
show_volcano(
    result.y_pred, protein_of_interest, other_protein, title_addendum="Pred "
)

result_y_pred = SingleChannelRegressionExperiment().run_adhoc_experiment(
    X,
    y[:, 0],
    SingleChannelRegressionRNN,
    model_save_name="best_ypred_single_channel_regression_model.h5",
    load_trained_model=True,
    optimizer=keras.optimizers.Adam(learning_rate=0.001)
)
result_fold_change = SingleChannelRegressionExperiment().run_adhoc_experiment(
    X,
    y[:, 1],
    SingleChannelRegressionRNN,
    model_save_name="best_fc_single_channel_regression_model.h5",
    optimizer=keras.optimizers.Adam(learning_rate=0.001)
)
pdb.set_trace()
show_volcano(y, protein_of_interest, other_protein, title_addendum="True ")
# show_volcano(
#     result.y_pred, protein_of_interest, other_protein, title_addendum="Pred "
# )
