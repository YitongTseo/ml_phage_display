import pathlib
import sys
HOME_DIRECTORY = pathlib.Path().absolute()
sys.path.append(str(HOME_DIRECTORY) + '/src')

from sklearn.metrics import confusion_matrix

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
