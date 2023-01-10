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
from analysis.shapley_additive_analysis import shapley_analysis

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
y_f = y[:,1]>0
y_p = y[:,0]

result = BinaryClassificationExperiment().run_adhoc_experiment(
    X, y_f, BinaryClassificationRNN, load_trained_model=True
)
# investigation_target = list(zip(X, peptides))[:1]
# shapley_analysis(result.trained_model, X, investigation_target)
investigation_target = list(zip(X, peptides))[:5]
shapley_analysis(result.trained_model, X, investigation_target)
# investigation_target = list(zip(X, peptides))[2:3]
# shapley_analysis(result.trained_model, X, investigation_target)
# investigation_target = list(zip(X, peptides))[3:4]
# shapley_analysis(result.trained_model, X, investigation_target)
# investigation_target = list(zip(X, peptides))[4:5]
# shapley_analysis(result.trained_model, X, investigation_target)
# investigation_target = list(zip(X, peptides))[5:6]
# shapley_analysis(result.trained_model, X, investigation_target)

# X, y = build_dataset(
#     R3_lib,
#     protein_of_interest=protein_of_interest,
#     other_protein=other_protein,
#     dataset_type=DATASET_TYPE.BINARY_CLASSIFICATION,
# )
# result = BinaryClassificationExperiment().run_adhoc_experiment(
#     X, y, BinaryClassificationRNN
# )


print("now what...")


# experiment = RegressionExperiment()
# result = experiment.run_adhoc_experiment(X, y, RegressionRNN, load_trained_model=False)
# show_volcano(
#     result.y_pred, protein_of_interest, other_protein, title_addendum="Predicted "
# )
# print("now what...")
