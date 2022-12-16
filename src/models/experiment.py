import pathlib
import sys
HOME_DIRECTORY = pathlib.Path().absolute()
sys.path.append(str(HOME_DIRECTORY) + '/src')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from typing import Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class Result:
    metrics: dict[str, Any] = None
    trained_model: Optional[Any] = None
    y_pred: Optional[np.array] = None
    y_true: Optional[np.array] = None


def train(
    X_train,
    y_train,
    model_architecture,
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
):
    def scheduler(epoch, lr):
        if epoch < 5:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    # TODO: Unsure if this LearningRateScheduler is actually hooked up correctly
    callback = keras.callbacks.LearningRateScheduler(scheduler)
    model = model_architecture(optimizer)
    model.fit(
        x=X_train,
        y=y_train,
        batch_size=128,
        epochs=16,
        verbose="auto",
        validation_split=0.1,
        initial_epoch=0,
        class_weight={1: 0.5, 0: 0.5},
        validation_freq=1,
        use_multiprocessing=False,
    )
    return model


def predict(model, X_test, y_true):
    y_pred = model(X_test)
    y_pred = y_pred.numpy().reshape(-1) >= 0.5
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    acc = (tn + tp) / (tn + fp + fn + tp)
    print(tn, fp, fn, tp)
    print("accuracy", acc)
    print("precision", pre)
    print("recall", rec)
    return (y_pred, acc, pre, rec)


def run_adhoc_experiment(X, y, model_architecture, test_train_split=0.2):
    (X_train, X_test, y_train, y_test) = train_test_split(
        X,
        y,
        test_size=test_train_split,
        shuffle=True,
        random_state=5,
    )
    model = train(
        X_train,
        y_train,
        model_architecture,
    )
    y_pred, acc, pre, rec = predict(model, X_test, y_test)
    return Result(
        trained_model=model,
        metrics={
            "accuracy": acc,
            "Precision": pre,
            "Recall": rec,
        },
        y_pred=y_pred,
        y_true=y_test,
    )

def run_cross_validation_experiment(X, y, model_architecture, n_splits=5):
    kf = KFold(n_splits=n_splits)
    results = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = train(
            X_train,
            y_train,
            model_architecture,
        )
        y_pred, acc, pre, rec = predict(model, X_test, y_test)
        results.append(
            Result(
                trained_model=model,
                metrics={
                    "accuracy": acc,
                    "Precision": pre,
                    "Recall": rec,
                },
                y_pred=y_pred,
                y_true=y_test,
            )
        )
    return results
