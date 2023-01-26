import pdb
from preprocessing.data_loading import (
    read_data_and_preprocess,
    build_dataset,
)
import numpy as np
import preprocessing.data_loading as data_loading
from models.rnn import BinaryClassificationRNN
import matplotlib.pyplot as plt
import preprocessing.y_representation as y_representation
from analysis.scatter_plots import show_volcano
from sklearn.model_selection import train_test_split
import models.experiment as experiment
import models.rnn as rnn
from tensorflow import keras

# R3_lib = read_data_and_preprocess(datafile="12ca5-MDM2-mCDH2-R3.csv")
# protein_of_interest = "12ca5" 
# other_protein = "MDM2"
# X, y, y_raw, peptides = data_loading.build_dataset(
#     R3_lib,
#     protein_of_interest=protein_of_interest,
#     other_protein=other_protein,
#     aa_representations=[
#         data_loading.AA_REPRESENTATION.PRO2VEC,
#         data_loading.AA_REPRESENTATION.RAA,
#         data_loading.AA_REPRESENTATION.PHYSIOCHEM_PROPERTIES,
#         data_loading.AA_REPRESENTATION.ONE_HOT,
#     ],
# )
# (X_train, X_test, y_train, y_test) = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     shuffle=True,
#     random_state=5,
# )
# model = experiment.BinaryClassificationExperiment().train(
#     X_train,
#     y_train,
#     rnn.Joint_BinaryClassificationRNN_gelu,
#     optimizer=keras.optimizers.Adam(learning_rate=0.001),
# )
# y_pred = model(X_test) > 0.5
# all_targets = list(zip(X, peptides))
# # First let's cherry pick a couple of peptides which we know match to see how the analysis goes.
# # BIG TODO(Yitong): Debug why the expected value does not match the sum of the values...
# # shapley_analysis(
# #     model,
# #     X,
# #     [x for x in all_targets if x[1] == "AICRGDYAC"], # Known binder
# #     investigation_type=INVESTIGATION_TYPE.BY_FEATURE,
# #     num_background_samples=200,
# #     num_perturbation_samples=2000,
# # )
# # shapley_analysis(
# #     model,
# #     X,
# #     [x for x in all_targets if x[1] == "DYPDYAE"], # Known binder
# #     investigation_type=INVESTIGATION_TYPE.BY_FEATURE,
# #     num_background_samples=200,
# #     num_perturbation_samples=2000,
# # )
# # shapley_analysis(
# #     model,
# #     X,
# #     [x for x in all_targets if x[1] == "ARCVGDYAC"], # Known binder
# #     investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
# #     num_background_samples=200,
# #     num_perturbation_samples=2000,
# # )
# # shapley_analysis(
# #     model,
# #     X,
# #     [x for x in all_targets if x[1] == "ACLYWWWCR"], # Known non-binder
# #     investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
# #     num_background_samples=200,
# #     num_perturbation_samples=2000,
# # )
# # shapley_analysis(
# #     model,
# #     X,
# #     [x for x in all_targets if x[1] == "ACVVIKSCF"], # Known non-binder
# #     investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
# #     num_background_samples=200,
# #     num_perturbation_samples=2000,
# # )

# # # Let's do an investigation of the first 300 proteins
# investigation_target = all_targets[:100]
# shapley_analysis(
#     model,
#     X,
#     investigation_target,
#     investigation_type=INVESTIGATION_TYPE.BY_FEATURE,
# )


""" -----------------LOOK INTO MDM2!---------------------------------------------------------------------
"""

R3_lib = read_data_and_preprocess(datafile="12ca5-MDM2-mCDH2-R3.csv")
protein_of_interest = "MDM2"#"MDM2" 
other_protein = "12ca5"
X, y, y_raw, peptides = data_loading.build_dataset(
    R3_lib,
    protein_of_interest=protein_of_interest,
    other_protein=other_protein,
    aa_representations=[
        data_loading.AA_REPRESENTATION.PRO2VEC,
        data_loading.AA_REPRESENTATION.RAA,
        data_loading.AA_REPRESENTATION.PHYSIOCHEM_PROPERTIES,
        data_loading.AA_REPRESENTATION.ONE_HOT,
    ],
)
(X_train, X_test, y_train, y_test) = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=True,
    random_state=5,
)
model = experiment.BinaryClassificationExperiment().train(
    X_train,
    y_train,
    rnn.Joint_BinaryClassificationRNN_gelu,
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    load_trained_model=False
)
y_pred = model(X_test) > 0.5
all_targets = list(zip(X, peptides))
# We have to load this after the dataset is built so we get the correct feature dimensions
from analysis.shapley_additive_analysis import shapley_analysis, INVESTIGATION_TYPE



# show_volcano(
#     y, protein_of_interest, other_protein, title_addendum="True "
# )
# y_f = y[:, 1] > 0
# y_p = y[:, 0]

# result = BinaryClassificationExperiment(epochs=16).run_adhoc_experiment(
#     X, y_f, BinaryClassificationRNN, #load_trained_model=True
# )
# all_targets = list(zip(X, peptides))
# # First let's cherry pick a couple of peptides which we know match to see how the analysis goes.
# # BIG TODO(Yitong): Debug why the expected value does not match the sum of the values...
# result.trained_model(np.array([[x for x in all_targets if x[1] == "AKCDFSWCM"][0][0]]))
# These are the targets which we learn are good...
learned_targets = [
    all_targets[idx]
    for idx in np.where(np.logical_and(y_pred[:,0], y_pred[:,1]))[0]
]
print('these are what the model predicts as true ', [p[1] for p in learned_targets])

# print('is ACWGWECIS a true sample? ', y[[idx for idx, x in enumerate(peptides) if x == "ACWGWECIS"][0]])
# shapley_analysis(
#     model,
#     X,
#     [x for x in all_targets if x[1] == "ACWGWECIS"], # Known binder
#     investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
#     num_background_samples=200,
#     num_perturbation_samples=2000,
# )
# print('is AFCVWRWCS a true sample? ', y[[idx for idx, x in enumerate(peptides) if x == "AFCVWRWCS"][0]])
# shapley_analysis(
#     model,
#     X,
#     [x for x in all_targets if x[1] == "AFCVWRWCS"], 
#     investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
#     num_background_samples=200,
#     num_perturbation_samples=2000,
# )
# shapley_analysis(
#     model,
#     X,
#     [x for x in all_targets if x[1] == "ALCQGDFNC"], # Known non-binder
#     investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
#     num_background_samples=200,
#     num_perturbation_samples=2000,
# )
# shapley_analysis(
#     model,
#     X,
#     [x for x in all_targets if x[1] == "AACWKHVCS"], # Known non-binder
#     investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
#     num_background_samples=200,
#     num_perturbation_samples=2000,
# )

# shapley_analysis(
#     model,
#     X,
#     [x for x in all_targets if x[1] == "ACVDYAACR"], # Known non-binder
#     investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
#     num_background_samples=200,
#     num_perturbation_samples=2000,
# )

# shapley_analysis(
#     model,
#     X,
#     [x for x in all_targets if x[1] == "AICRGDYAC"], # Known 12ca5 binder
#     investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
#     num_background_samples=200,
#     num_perturbation_samples=2000,
# )
# shapley_analysis(
#     model,
#     X,
#     [x for x in all_targets if x[1] == "DYPDYAE"], # Known 12ca5 binder
#     investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
#     num_background_samples=200,
#     num_perturbation_samples=2000,
# )

# Let's do an investigation of the first 300 proteins
investigation_target = all_targets[:100]
shapley_analysis(
    model,
    X,
    investigation_target,
    investigation_type=INVESTIGATION_TYPE.BY_FEATURE,
)
