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
    


