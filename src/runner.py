import pdb
from preprocessing.data_loading import read_data_and_preprocess, build_dataset
from models.experiment import (
    RegressionExperiment,
    BinaryClassificationExperiment,
)
from models.rnn import RegressionRNN
import matplotlib.pyplot as plt
import preprocessing.y_representation as y_representation


R3_lib = read_data_and_preprocess(datafile="12ca5-MDM2-mCDH2-R3.csv")

# def show_volcano(lib, protein_of_interest="MDM2", other_protein="12ca5"):
#     logfold = y_representation.calculate_log_fold_across_proteins(
#         lib,
#         protein_of_interest=protein_of_interest,
#         other_proteins=[
#             other_protein,
#         ],
#     )
#     pvalue = -y_representation.calculate_log_p_value(
#         lib, protein_of_interest=protein_of_interest, other_protein=other_protein
#     )
#     plt.title(protein_of_interest + " vs " + other_protein)
#     plt.ylabel("- p-value")
#     plt.xlabel("log fold")
#     plt.scatter(logfold, pvalue, alpha=0.2)
#     plt.show()
# show_volcano(R3_lib, protein_of_interest="MDM2", other_protein="12ca5")
# show_volcano(R3_lib, protein_of_interest="MDM2", other_protein="mCDH2")


def show_volcano(y, protein_of_interest, other_protein, title_addendum=""):
    plt.title(
        title_addendum
        + protein_of_interest
        + " vs "
        + other_protein
        + "\n(normalized by mean & std. dev)"
    )
    plt.ylabel("- p-value")
    plt.xlabel("log fold")
    plt.scatter(y[:, 1], -y[:, 0], alpha=0.2)
    plt.show()


protein_of_interest = "MDM2"
other_protein = "12ca5"
X, y = build_dataset(
    R3_lib, protein_of_interest=protein_of_interest, other_protein=other_protein
)

experiment = RegressionExperiment()
result = experiment.run_adhoc_experiment(X, y, RegressionRNN, load_trained_model=False)
show_volcano(
    result.y_pred, protein_of_interest, other_protein, title_addendum="Predicted "
)
print("now what...")
