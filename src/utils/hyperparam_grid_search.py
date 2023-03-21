
import sys
sys.path.append("src/")
import operator
from importlib import reload
import src.preprocessing.data_loading as data_loading
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
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from analysis.heatmap_analysis import generate_heatmap
from analysis.hit_rate_analysis import (
    plot_ratio_by_ranking,
    sort_peptides_by_model_ranking,
)
import json
import pdb
import itertools
from functools import partial
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval, Trials

protein_of_interest = "MDM2"
other_protein = "12ca5"

er_lib = data_loading.read_data_and_preprocess(
    datafile=f"{protein_of_interest}_merged_ER.csv"
)
fc_pval_lib = data_loading.read_data_and_preprocess(datafile="12ca5-MDM2-mCDH2-R3.csv")
joint_lib = er_lib.merge(
    right=fc_pval_lib, right_on="Peptide", left_on="Peptide", suffixes=("x", "")
)
X, y_raw, peptides = data_loading.build_joint_dataset(
    joint_lib, protein_of_interest=protein_of_interest, other_protein=other_protein
)

(X_eval, X, y_eval, y_raw, peptides_eval, peptides) = train_test_split(
    X,
    y_raw,
    peptides,
    test_size=0.8,
    shuffle=True,
    random_state=5,
)


hyper_dict = {
    "width": [64], 
    "depth": [ 9 ], 
    "batch_size": [64, 128],
    "weight_decay": [0.05, 0.1, 0.3, 0.5, 0.8],
    "learning_rate": [0.0001, 0.0005],
    "num_epochs": [1, 3, 5, 7, 10],
    "dropout": [0.05, 0.1, 0.3, 0.5, 0.8],
}
parameter_space =  { "width": hp.choice("width", hyper_dict['width']),
    "depth": hp.choice("depth", hyper_dict['depth']), 
    "batch_size": hp.choice("batch_size", hyper_dict['batch_size']), 
    "weight_decay": hp.choice("weight_decay", hyper_dict['weight_decay']), 
    "learning_rate": hp.choice("learning_rate", hyper_dict['learning_rate']),
    "num_epochs": hp.choice("num_epochs", hyper_dict['num_epochs']),
    "dropout": hp.choice("dropout", hyper_dict['dropout']),
}


def hyperoptoutput2param(best):
    '''Change hyperopt output to dictionary with values '''
    for key in best.keys():
        if key in hyper_dict.keys():
            best[key] = hyper_dict[key][ best[key] ] 
            
    return best


results = []
def model_eval(permutation):
    def benchmark(mdm2_result, X, peptides, pred_ranking):
        mdm2_model = mdm2_result.trained_model
        return plot_ratio_by_ranking(
            peptides=peptides,
            ca5_y_ranking=None,
            mdm2_y_ranking=[pred_ranking(pred) for pred in mdm2_model(X)],
            title="",
            plot_theoretical_maximums=True,
            plot=False
        )

    mdm2_results, yscaler = experiment.Experiment().run_cross_validation_experiment(
        X,
        y_raw,
        partial(
            rnn.ThreeChannelRegressionRNN_gelu, depth=permutation["depth"], width=permutation["width"], dropout=permutation['dropout']
        ),
        model_save_name=f"grid_search_models/learning_rate({permutation['learning_rate']})_batch_size({permutation['batch_size']})_dropout({permutation['dropout']})_depth({permutation['depth']})_width({permutation['width']})_weightdecay({permutation['weight_decay']}).h5",
        normalize=True,
        n_splits=3,
        batch_size=permutation["batch_size"],
        optimizer=partial(keras.optimizers.Adam, learning_rate=permutation['learning_rate'], weight_decay=permutation["weight_decay"]), 
        num_epochs=permutation['num_epochs']
    )

    permutation['all_combined_auc'] = [benchmark(mdm2_result, X, peptides, lambda x: x[0] + x[1] + x[2]) for mdm2_result in mdm2_results]
    permutation['all_just_er_auc'] = [benchmark(mdm2_result, X, peptides, lambda x: x[2]) for mdm2_result in mdm2_results]
    permutation['all_just_pval_auc'] = [benchmark(mdm2_result, X, peptides, lambda x: x[0]) for mdm2_result in mdm2_results]
    permutation['all_just_fc_auc'] = [benchmark(mdm2_result, X, peptides, lambda x: x[1]) for mdm2_result in mdm2_results]

    permutation['combined_auc'] = np.max(permutation['all_combined_auc'])
    permutation['just_er_auc'] = np.max(permutation['all_just_er_auc'])
    permutation['just_pval_auc'] = np.max(permutation['all_just_pval_auc'])
    permutation['just_fc_auc'] = np.max(permutation['all_just_fc_auc'])

    results.append(permutation)
    print(results)
    json.dump( results, open( "hyperparam_output_focused_on_regularization.json", 'w' ) )
    return 1 -permutation['combined_auc']
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# breakpoint()
# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()


trials = Trials()
best = fmin(model_eval, parameter_space, algo=tpe.suggest, max_evals=80, trials=trials) # this will take a while to run 
best = hyperoptoutput2param(best)
print('best! ', best)

# keys, values = zip(*hyper_dict.items())
# permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
# results = []
# for permutation in permutations_dicts:
#     mdm2_results = experiment.Experiment().run_cross_validation_experiment(
#         X,
#         y_raw,
#         partial(
#             rnn.ThreeChannelRegressionRNN_gelu, depth=permutation["depth"], width=permutation["width"]
#         ),
#         model_save_name=f"grid_search_models/learning_rate({permutation['learning_rate']})_batch_size({permutation['batch_size']})_depth({permutation['depth']})_width({permutation['width']})_weightdecay({permutation['weight_decay']}).h5",
#         normalize=True,
#         n_splits=2,
#         batch_size=permutation["batch_size"],
#         optimizer=keras.optimizers.Adam(learning_rate=permutation['learning_rate'], weight_decay=permutation["weight_decay"]), 
#         num_epochs=1
#     )

#     permutation['combined_auc'] = np.max([benchmark(mdm2_result, X, peptides, lambda x: x[0] + x[1] + x[2]) for mdm2_result in mdm2_results])
#     permutation['just_er_auc'] = np.max([benchmark(mdm2_result, X, peptides, lambda x: x[2]) for mdm2_result in mdm2_results])
#     permutation['just_pval_auc'] = np.max([benchmark(mdm2_result, X, peptides, lambda x: x[0]) for mdm2_result in mdm2_results])
#     permutation['just_fc_auc'] = np.max([benchmark(mdm2_result, X, peptides, lambda x: x[1]) for mdm2_result in mdm2_results])

#     permutation['all_combined_auc'] = [benchmark(mdm2_result, X, peptides, lambda x: x[0] + x[1] + x[2]) for mdm2_result in mdm2_results]
#     permutation['all_just_er_auc'] = [benchmark(mdm2_result, X, peptides, lambda x: x[2]) for mdm2_result in mdm2_results]
#     permutation['all_just_pval_auc'] = [benchmark(mdm2_result, X, peptides, lambda x: x[0]) for mdm2_result in mdm2_results]
#     permutation['all_just_fc_auc'] = [benchmark(mdm2_result, X, peptides, lambda x: x[1]) for mdm2_result in mdm2_results]

#     results.append(permutation)
#     print(results)
#     json.dump( results, open( "hyperparam_output.json", 'w' ) )


# # (
#     (
#         mdm2_X_train,
#         mdm2_X_test,
#         mdm2_y_train,
#         mdm2_y_test,
#         mdm2_peptides_train,
#         mdm2_peptides_test,
#     ),
#     mdm2_results,
# ) = experiment.Experiment().run_adhoc_experiment(
#     X,
#     y_raw,
#     partial(
#         rnn.ThreeChannelRegressionRNN_gelu, param_power=permutation["param_power"]
#     ),
#     optimizer=keras.optimizers.Adam(
#         learning_rate=0.001, weight_decay=permutation["weight_decay"]
#     ),
#     load_trained_model=False,
#     model_save_name=f"grid_search_models/mdm2_three_channel_GCV_parampower_deeper({permutation['param_power']})_weightdecay({permutation['weight_decay']}).h5",
#     other_datasets=[
#         peptides,
#     ],
#     normalize=True,
# )

# # %%
# mdm2_model.count_params()


# # %%
# mdm2_model = mdm2_results.trained_model
# mdm2_pred = mdm2_model(X)
# mdm2_ordering = []
# for pred in mdm2_pred.numpy():
#     mdm2_ordering.append(pred[2] + pred[0] + pred[1])

# plot_ratio_by_ranking(
#     peptides=peptides,#mdm2_peptides_test,
#     ca5_y_ranking=None,
#     mdm2_y_ranking=mdm2_ordering,#np.squeeze(mdm2_only_er_results.y_pred), #mdm2_ordering,
#     title="Hit Rates Against Enrichment Ratio BiLSTM Rankings \n(On Same Held Out Test Set)",
#     plot_theoretical_maximums=True,
# )


# # %% [markdown]
# # # Parity plots look poor

# # %%
# import seaborn as sns
# # sns.jointplot(x=mdm2_y_test[:, 1], y=mdm2_y_test[:, 2],kind='hex')
# sns.jointplot(x=y_raw[:, 2], y=mdm2_pred[:, 2], kind='hex')
# plt.xlabel('True ER')
# plt.ylabel('Pred ER')
# plt.title('True versus Pred ER')

# # %%
# mdm2_pred.numpy()[:10].mean(axis=0)

# # %%
# import seaborn as sns

# def plot_relations(x_idx, y_idx, mdm2_ordering, mdm2_pred=mdm2_pred, vals=['Pval','FC', 'ER']):
#     sns.jointplot(x=mdm2_pred[:, x_idx], y=mdm2_pred[:, y_idx]) # , kind='hex'
#     plt.xlabel('Pred ' + vals[x_idx])
#     plt.ylabel('Pred '+ vals[y_idx])
#     plt.title('Predicted vals')

#     top_mdm2_mask = mdm2_ordering >= np.partition(mdm2_ordering, kth=-500)[-500]
#     plt.scatter(
#         x=mdm2_pred[:, x_idx][top_mdm2_mask],
#         y=mdm2_pred[:, y_idx][top_mdm2_mask],
#         color="red",
#         alpha=0.01,
#     )

# mdm2_ordering = []
# for pred in mdm2_pred.numpy():
#     mdm2_ordering.append(pred[2]) # pred[0] + pred[1] + 
# plot_relations(1, 0, mdm2_ordering=mdm2_ordering)

# # %%

# from analysis.hit_rate_analysis import (
#     plot_ratio_by_ranking,
#     sort_peptides_by_model_ranking,
#     seq_contains_mdm2_motif,
# )

# # this is just the hypothetical best ordering
# hypothetical_best_mdm2_ordering = np.array([ 1.0 if seq_contains_mdm2_motif(pep) else 0.0 for pep in peptides])
# plot_relations(1, 0, mdm2_ordering=hypothetical_best_mdm2_ordering)

# # %%


# # %%


# # %%


# # %%


# # %%


# # %%


# # %%


# # %%


# # %%
# protein_of_interest = "MDM2"
# other_protein = "12ca5"

# er_lib = data_loading.read_data_and_preprocess(
#     datafile=f"{protein_of_interest}_merged_ER.csv"
# )
# fc_pval_lib = data_loading.read_data_and_preprocess(datafile="12ca5-MDM2-mCDH2-R3.csv")
# joint_lib = er_lib.merge(
#     right=fc_pval_lib, right_on="Peptide", left_on="Peptide", suffixes=("x", "")
# )
# X, y_raw, peptides = data_loading.build_joint_dataset(
#     joint_lib, protein_of_interest=protein_of_interest, other_protein=other_protein
# )
# (
#     (
#         mdm2_X_train,
#         mdm2_X_test,
#         mdm2_y_er_train,
#         mdm2_y_er_test,
#         mdm2_peptides_train,
#         mdm2_peptides_test,
#     ),
#     mdm2_only_er_results,
# ) = experiment.Experiment().run_adhoc_experiment(
#     X,
#     y_raw[:, 2].reshape(-1,1),
#     rnn.SingleRegressionRNN_gelu, #rnn.SingleChannelRegressionRNN, #rnn.ThreeChannelRegressionRNN,
#     optimizer=keras.optimizers.Adam(learning_rate=0.001),
#     load_trained_model=True,
#     model_save_name="mdm_2_ER_only_with_no_cadherin_data_2.h5",#'mdm_2_ER_3.h5',
#     other_datasets=[
#         peptides,
#     ],
#     normalize=True
# )
# mdm2_er_model = mdm2_only_er_results.trained_model
# only_er_volcano_mdm2_ordering = np.squeeze([pred_val for pred_val in mdm2_er_model(X)])
# print(X.shape)
# mdm2_ordering = only_er_volcano_mdm2_ordering

# plot_ratio_by_ranking(
#     peptides=peptides,#mdm2_peptides_test,
#     ca5_y_ranking=None,
#     mdm2_y_ranking=mdm2_ordering,#np.squeeze(mdm2_only_er_results.y_pred), #mdm2_ordering,
#     title="Hit Rates Against Enrichment Ratio BiLSTM Rankings \n(On All Samples)",
#     plot_theoretical_maximums=True,
# )


# # %%

# from analysis.hit_rate_analysis import (
#     plot_ratio_by_ranking,
#     sort_peptides_by_model_ranking,
#     seq_contains_mdm2_motif,
# )

# # this is just the hypothetical best ordering
# hypothetical_best_mdm2_ordering = np.array([ 1.0 if seq_contains_mdm2_motif(pep) else 0.0 for pep in peptides])
# plot_relations(1, 0, mdm2_ordering=np.squeeze(mdm2_er_model(X).numpy()))


