import pdb
from preprocessing.data_loading import (
    read_data_and_preprocess,
    build_dataset,
    DATASET_TYPE,
)
from models.experiment import (
    RegressionExperiment,
    BinaryClassificationExperiment,
)
from models.rnn import RegressionRNN, BinaryClassificationRNN
import matplotlib.pyplot as plt
import preprocessing.y_representation as y_representation
from analysis.shapley_additive_analysis import shapley_analysis, INVESTIGATION_TYPE

R3_lib = read_data_and_preprocess(datafile="12ca5-MDM2-mCDH2-R3.csv")
protein_of_interest = "12ca5"
other_protein = "MDM2"
X, y, peptides = build_dataset(
    R3_lib,
    protein_of_interest=protein_of_interest,
    other_protein=other_protein,
    # Huh JOINT_REGRESSION for the y building.
    dataset_type=DATASET_TYPE.JOINT_REGRESSION,
)
y_f = y[:, 1] > 0
y_p = y[:, 0]

result = BinaryClassificationExperiment().run_adhoc_experiment(
    X, y_f, BinaryClassificationRNN, load_trained_model=True
)
all_targets = list(zip(X, peptides))
# First let's cherry pick a couple of peptides which we know match to see how the analysis goes.
# BIG TODO(Yitong): Debug why the expected value does not match the sum of the values...
# shapley_analysis(
#     result.trained_model,
#     X,
#     [x for x in all_targets if x[1] == "AICRGDYAC"], # Known binder
#     investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
#     num_background_samples=200,
#     num_perturbation_samples=2000,
# )
# shapley_analysis(
#     result.trained_model,
#     X,
#     [x for x in all_targets if x[1] == "DYPDYAE"], # Known binder
#     investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
#     num_background_samples=200,
#     num_perturbation_samples=2000,
# )
# shapley_analysis(
#     result.trained_model,
#     X,
#     [x for x in all_targets if x[1] == "ARCVGDYAC"], # Known binder
#     investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
#     num_background_samples=200,
#     num_perturbation_samples=2000,
# )
# shapley_analysis(
#     result.trained_model,
#     X,
#     [x for x in all_targets if x[1] == "ACLYWWWCR"], # Known non-binder
#     investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
#     num_background_samples=200,
#     num_perturbation_samples=2000,
# )
# shapley_analysis(
#     result.trained_model,
#     X,
#     [x for x in all_targets if x[1] == "ACVVIKSCF"], # Known non-binder
#     investigation_type=INVESTIGATION_TYPE.BY_AMINO_ACID,
#     num_background_samples=200,
#     num_perturbation_samples=2000,
# )

# Let's do an investigation of the first 300 proteins
investigation_target = all_targets[:200]
shapley_analysis(
    result.trained_model,
    X,
    investigation_target,
    investigation_type=INVESTIGATION_TYPE.BY_FEATURE,
)
