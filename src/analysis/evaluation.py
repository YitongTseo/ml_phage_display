import pathlib
import sys
HOME_DIRECTORY = pathlib.Path().absolute()
sys.path.append(str(HOME_DIRECTORY) + '/src')
import sklearn.metrics as metrics

from sklearn.metrics import confusion_matrix


def classification_evaluation_joint(y_true, y_pred):
    # y_true & y_test shape (# samples, 2) for both p-value & fc
    print('P-VAL')
    tn, fp, fn, tp = confusion_matrix(y_true[:, 0], y_pred[:, 0]).ravel()
    print('\t accuracy: ', metrics.accuracy_score(y_true[:, 0], y_pred[:, 0]))
    print('\t precision: ', metrics.precision_score(y_true[:, 0], y_pred[:, 0]))
    print('\t recall: ', metrics.recall_score(y_true[:, 0], y_pred[:, 0]))
    print('\t # neg predictions: ', tn + fn)
    print('\t # pos predictions: ', tp + fp)

    print('Log Fold')
    tn, fp, fn, tp = confusion_matrix(y_true[:, 1], y_pred[:, 1]).ravel()
    print('\t accuracy: ', metrics.accuracy_score(y_true[:, 1], y_pred[:, 1]))
    print('\t precision: ', metrics.precision_score(y_true[:, 1], y_pred[:, 1]))
    print('\t recall: ', metrics.recall_score(y_true[:, 1], y_pred[:, 1]))
    print('\t # neg predictions: ', tn + fn)
    print('\t # pos predictions: ', tp + fp)


def classifcation_evaluation(y_train, y_pred, y_cutoff):
    y_train = y_train>y_cutoff
    y_pred = y_pred >0.5
    tn, fp, fn, tp = confusion_matrix(y_train>0, y_pred).ravel()
    acc, pre, rec = [], [], []
    print(tn, fp, fn, tp)
    print("accuracy", (tn+tp)/(tn+fp+fn+tp))
    acc.append((tn+tp)/(tn+fp+fn+tp))
    print("precision", tp/(tp+fp))
    pre.append(tp/(tp+fp))
    print("recall", tp/(tp+fn))
    rec.append(tp/(tp+fn))
    

def single_regression_evaluation(y_train, y_pred, y_cutoff):
    y_train = y_train>y_cutoff
    y_pred = y_pred >0.5
    tn, fp, fn, tp = confusion_matrix(y_train>0, y_pred).ravel()
    acc, pre, rec = [], [], []
    print(tn, fp, fn, tp)
    print("accuracy", (tn+tp)/(tn+fp+fn+tp))
    acc.append((tn+tp)/(tn+fp+fn+tp))
    print("precision", tp/(tp+fp))
    pre.append(tp/(tp+fp))
    print("recall", tp/(tp+fn))
    rec.append(tp/(tp+fn))


def joint_regression_evaluation(y1_train, y1_pred, y1_cutoff, y2_train, y2_pred, y2_cutoff):
    y1_train = y1_train>y1_cutoff
    y2_train = y2_train>y2_cutoff
    y1_pred = y1_pred >0.5
    y2_pred = y2_pred >0.5
    tn, fp, fn, tp = confusion_matrix((y1_train)*(y2_train), (y1_pred)*(y2_pred)).ravel()
    acc, pre, rec = [], [], []
    print(tn, fp, fn, tp)
    print("accuracy", (tn+tp)/(tn+fp+fn+tp))
    acc.append((tn+tp)/(tn+fp+fn+tp))
    print("precision", tp/(tp+fp))
    pre.append(tp/(tp+fp))
    print("recall", tp/(tp+fn))
    rec.append(tp/(tp+fn))
