import sys

import pdb

import operator
from importlib import reload
import preprocessing.data_loading as data_loading
import models.experiment as experiment
import models.rnn as rnn
import analysis.umap_analysis as umap
import analysis.scatter_plots as scatter_plots
import analysis.evaluation as evaluation
from preprocessing.X_representation import RAA_to_seq
from preprocessing.X_representation_utils import initialize_Proc2Vec_embeddings
from utils.utils import find_3mer
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import pdb
from sklearn.model_selection import train_test_split
from analysis.model_run_comparisons import model_comparison_matrix_visualization

# model_comparison_matrix_visualization()


R3_lib = data_loading.read_data_and_preprocess(datafile="12ca5-MDM2-mCDH2-R3.csv")
protein_of_interest = "MDM2"
other_protein = "12ca5"
X, y, peptides = data_loading.build_dataset(
    R3_lib,
    protein_of_interest=protein_of_interest,
    other_protein=other_protein,
    aa_representations = [
        data_loading.AA_REPRESENTATION.PRO2VEC,
        data_loading.AA_REPRESENTATION.RAA,
        data_loading.AA_REPRESENTATION.PHYSIOCHEM_PROPERTIES,
        data_loading.AA_REPRESENTATION.ONE_HOT,
    ]
)
sns.histplot(y[:,0])
plt.show()
# TODO(Yitong): are these the right cutoffs???... the yplot shows the same...
y_f_cutoff = -0.6565247891915524 #0.9419173905476301
y_p_cutoff = -0.44638568713238386 #-0.4463856871323837

#  -0.44638568713238386, and log FC value cutoff is -0.6565247891915524
# TODO: Move these into data_loading...
y[:, 1] = y[:, 1] > y_f_cutoff
y[:, 0] = y[:, 0] > y_p_cutoff

pdb.set_trace()

result = experiment.BinaryClassificationExperiment().run_adhoc_experiment(
    X, y, rnn.Joint_BinaryClassificationRNN_gelu, load_trained_model=False
)
model = result.trained_model

(
    X_train,
    X_test,
    y_f_train,
    y_f_test,
    y_p_train,
    y_p_test,
) = scatter_plots.train_test_split_sample(X, y)


print("Evaluation on training data")
print("\tLog Fold")
evaluation.classifcation_evaluation(y_f_train, model(X_train)[:, 1], y_f_cutoff)
print("\tLog P Value")
evaluation.classifcation_evaluation(y_p_train, model(X_train)[:, 0], y_p_cutoff)

print("\nFold evaluation on test data")
print("\tLog Fold")
evaluation.classifcation_evaluation(y_f_test, model(X_test)[:, 1], y_f_cutoff)
print("\tLog P Value")
evaluation.classifcation_evaluation(y_p_test, model(X_test)[:, 0], y_p_cutoff)
embedding = umap.embedding_classification(model, X_train)
pdb.set_trace()
print("how are we doing?")

# show_volcano(y, protein_of_interest, other_protein, title_addendum="True ")
# show_volcano(result.y_pred, protein_of_interest, other_protein, title_addendum="Pred ")

# result_y_pred = SingleChannelRegressionExperiment().run_adhoc_experiment(
#     X,
#     y[:, 0],
#     SingleChannelRegressionRNN,
#     model_save_name="best_ypred_single_channel_regression_model.h5",
#     load_trained_model=True,
#     optimizer=keras.optimizers.Adam(learning_rate=0.001)
# )
# result_fold_change = SingleChannelRegressionExperiment().run_adhoc_experiment(
#     X,
#     y[:, 1],
#     SingleChannelRegressionRNN,
#     model_save_name="best_fc_single_channel_regression_model.h5",
#     optimizer=keras.optimizers.Adam(learning_rate=0.001)
# )
# pdb.set_trace()
# show_volcano(y, protein_of_interest, other_protein, title_addendum="True ")
# (X_train, X_test, y_train, y_test) = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     shuffle=True,
#     random_state=5,
# )
# embedding = embedding_regression(result.trained_model, X_test)
# pdb.set_trace()
# UMAP_log_P(embedding, y_test[:, 0])
# UMAP_log_Fold(embedding, y_test[:, 1])

# show_volcano(
#     result.y_pred, protein_of_interest, other_protein, title_addendum="Pred "
# )
