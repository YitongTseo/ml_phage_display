import pdb
from preprocessing.data_loading import (
    read_data_and_preprocess,
    build_dataset,
    DATASET_TYPE,
)
import numpy as np
from models.experiment import (
    TwoChannelRegressionExperiment,
    BinaryClassificationExperiment,
)
from models.rnn import BinaryClassificationRNN
import matplotlib.pyplot as plt
import preprocessing.y_representation as y_representation
from analysis.shapley_additive_analysis import shapley_analysis, INVESTIGATION_TYPE
from analysis.scatter_plots import show_volcano

# R3_lib = read_data_and_preprocess(datafile="12ca5-MDM2-mCDH2-R3.csv")
# protein_of_interest = "12ca5"
# other_protein = "MDM2"
# X, y, peptides = build_dataset(
#     R3_lib,
#     protein_of_interest=protein_of_interest,
#     other_protein=other_protein,
#     # Huh JOINT_REGRESSION for the y building.
#     dataset_type=DATASET_TYPE.JOINT_REGRESSION,
# )
# y_f = y[:, 1] > 0
# y_p = y[:, 0]

# result = BinaryClassificationExperiment().run_adhoc_experiment(
#     X, y_f, BinaryClassificationRNN, #load_trained_model=True
# )
# all_targets = list(zip(X, peptides))
# pdb.set_trace()
# # First let's cherry pick a couple of peptides which we know match to see how the analysis goes.
# # BIG TODO(Yitong): Debug why the expected value does not match the sum of the values...
# # shapley_analysis(
# #     result.trained_model,
# #     X,
# #     [x for x in all_targets if x[1] == "AICRGDYAC"], # Known binder
# #     investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
# #     num_background_samples=200,
# #     num_perturbation_samples=2000,
# # )
# # shapley_analysis(
# #     result.trained_model,
# #     X,
# #     [x for x in all_targets if x[1] == "DYPDYAE"], # Known binder
# #     investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
# #     num_background_samples=200,
# #     num_perturbation_samples=2000,
# # )
# # shapley_analysis(
# #     result.trained_model,
# #     X,
# #     [x for x in all_targets if x[1] == "ARCVGDYAC"], # Known binder
# #     investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
# #     num_background_samples=200,
# #     num_perturbation_samples=2000,
# # )
# # shapley_analysis(
# #     result.trained_model,
# #     X,
# #     [x for x in all_targets if x[1] == "ACLYWWWCR"], # Known non-binder
# #     investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
# #     num_background_samples=200,
# #     num_perturbation_samples=2000,
# # )
# # shapley_analysis(
# #     result.trained_model,
# #     X,
# #     [x for x in all_targets if x[1] == "ACVVIKSCF"], # Known non-binder
# #     investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
# #     num_background_samples=200,
# #     num_perturbation_samples=2000,
# # )

# # Let's do an investigation of the first 300 proteins
# investigation_target = all_targets[:100]
# shapley_analysis(
#     result.trained_model,
#     X,
#     investigation_target,
#     investigation_type=INVESTIGATION_TYPE.BY_FEATURE,
# )


""" -----------------LOOK INTO MDM2!---------------------------------------------------------------------
"""

R3_lib = read_data_and_preprocess(datafile="12ca5-MDM2-mCDH2-R3.csv")
# protein_of_interest = "12ca5"
# other_protein = "MDM2"

protein_of_interest = "MDM2"
other_protein = "12ca5"
X, y, peptides = build_dataset(
    R3_lib,
    protein_of_interest=protein_of_interest,
    other_protein=other_protein,
    # Huh JOINT_REGRESSION for the y building.
    dataset_type=DATASET_TYPE.BINARY_CLASSIFICATION,
)
# show_volcano(
#     y, protein_of_interest, other_protein, title_addendum="True "
# )
# y_f = y[:, 1] > 0
# y_p = y[:, 0]

result = BinaryClassificationExperiment(epochs=16).run_adhoc_experiment(
    X, y, BinaryClassificationRNN, load_trained_model=True
)
all_targets = list(zip(X, peptides))
# First let's cherry pick a couple of peptides which we know match to see how the analysis goes.
# BIG TODO(Yitong): Debug why the expected value does not match the sum of the values...
# result.trained_model(np.array([[x for x in all_targets if x[1] == "AKCDFSWCM"][0][0]]))
pdb.set_trace()

# These are the targets which we learn are good...
learned_targets = [
    all_targets[idx]
    for idx in np.where(result.y_pred == True)[0]
]
print('these are what the model predicts as true ', [p[1] for p in learned_targets])
print('is AQCLWSWCV a true sample? ', y[[idx for idx, x in enumerate(peptides) if x == "AQCLWSWCV"][0]])
shapley_analysis(
    result.trained_model,
    X,
    [x for x in all_targets if x[1] == "AQCLWSWCV"], 
    investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
    num_background_samples=200,
    num_perturbation_samples=2000,
)
print('is ACWGWECIS a true sample? ', y[[idx for idx, x in enumerate(peptides) if x == "ACWGWECIS"][0]])
shapley_analysis(
    result.trained_model,
    X,
    [x for x in all_targets if x[1] == "ACWGWECIS"], # Known binder
    investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
    num_background_samples=200,
    num_perturbation_samples=2000,
)
print('is AFCVWRWCS a true sample? ', y[[idx for idx, x in enumerate(peptides) if x == "AFCVWRWCS"][0]])
shapley_analysis(
    result.trained_model,
    X,
    [x for x in all_targets if x[1] == "AFCVWRWCS"], 
    investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
    num_background_samples=200,
    num_perturbation_samples=2000,
)
shapley_analysis(
    result.trained_model,
    X,
    [x for x in all_targets if x[1] == "ALCQGDFNC"], # Known non-binder
    investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
    num_background_samples=200,
    num_perturbation_samples=2000,
)
shapley_analysis(
    result.trained_model,
    X,
    [x for x in all_targets if x[1] == "AACWKHVCS"], # Known non-binder
    investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
    num_background_samples=200,
    num_perturbation_samples=2000,
)

shapley_analysis(
    result.trained_model,
    X,
    [x for x in all_targets if x[1] == "ACVDYAACR"], # Known non-binder
    investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
    num_background_samples=200,
    num_perturbation_samples=2000,
)

# Let's do an investigation of the first 300 proteins
investigation_target = all_targets[:100]
shapley_analysis(
    result.trained_model,
    X,
    investigation_target,
    investigation_type=INVESTIGATION_TYPE.BY_FEATURE,
)
