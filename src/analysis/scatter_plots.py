import pathlib
import sys
HOME_DIRECTORY = pathlib.Path().absolute()
sys.path.append(str(HOME_DIRECTORY) + '/src')

import umap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

def scatter_hist(x, y, ax, ax_histx, ax_histy, alpha=0.1):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, alpha=alpha)

    # now determine nice limits by hand:
    binwidth = 0.1
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

def show_volcano(y, protein_of_interest, other_protein, title_addendum=""):
    # Create a Figure, which doesn't have to be square.
    fig = plt.figure(constrained_layout=True)
    # Create the main axes, leaving 25% of the figure space at the top and on the
    # right to position marginals.
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    # The main axes' aspect can be fixed.
    ax.set(aspect=1)
    # Create marginal axes, which have 25% of the size of the main axes.  Note that
    # the inset axes are positioned *outside* (on the right and the top) of the
    # main axes, by specifying axes coordinates greater than 1.  Axes coordinates
    # less than 0 would likewise specify positions on the left and the bottom of
    # the main axes.
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    # Draw the scatter plot and marginals.
    scatter_hist(y[:, 1], y[:, 0], ax, ax_histx, ax_histy)
    ax.set_ylabel("- p-value")
    ax.set_xlabel("log fold")
    ax.set_title(
        title_addendum
        + protein_of_interest
        + " vs "
        + other_protein
        + "\n(normalized by mean & std. dev)"
    )
    plt.show()


def Kfold_sample(n_splits, stop_split, X, y): 
    kf = KFold(n_splits=n_splits)
    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_f_train, y_f_test = y[:,1][train_index], y[:,1][test_index]
        y_p_train, y_p_test = y[:,0][train_index], y[:,0][test_index]

        if i==stop_split:
            break
        i += 1  
    return X_train, X_test, y_f_train, y_f_test, y_p_train, y_p_test


def Predicted_vs_actual_plot(y_true, y_train, color, y_true_label, y_train_label, color_label):
    plt.xlabel("true {} value".format(y_true_label))
    plt.ylabel("predicted {} value".format(y_train_label))
    plt.scatter(y_true, y_train, c=color, cmap='bwr',alpha=0.2)
    plt.title("label by {} value".format(color_label))
    plt.colorbar()
    plt.show()
    


