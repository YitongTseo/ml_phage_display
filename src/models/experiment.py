import pathlib
import sys

HOME_DIRECTORY = pathlib.Path().absolute()
sys.path.append(str(HOME_DIRECTORY) + "/src")

import pdb
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from typing import Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from models.rnn import two_channel_mse, p_value_rmse, fold_rmse
from tensorflow.keras.metrics import mean_squared_error


@dataclass
class Result:
    metrics: dict[str, Any] = None
    trained_model: Optional[Any] = None
    y_pred: Optional[np.array] = None
    y_true: Optional[np.array] = None


class Experiment:
    def run_adhoc_experiment(
        self,
        X,
        y,
        model_architecture,
        test_train_split=0.2,
        load_trained_model=False,
        model_save_name=None,
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
    ):
        (X_train, X_test, y_train, y_test) = train_test_split(
            X,
            y,
            test_size=test_train_split,
            shuffle=True,
            random_state=5,
        )
        model = self.train(
            X_train,
            y_train,
            model_architecture,
            load_trained_model=load_trained_model,
            model_save_name=model_save_name,
            optimizer=optimizer,
        )
        y_pred, metrics = self.predict(model, X_test, y_test)
        return Result(
            trained_model=model,
            metrics=metrics,
            y_pred=y_pred,
            y_true=y_test,
        )

    def run_cross_validation_experiment(self, X, y, model_architecture, n_splits=5):
        kf = KFold(n_splits=n_splits)
        results = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = self.train(
                X_train,
                y_train,
                model_architecture,
            )
            y_pred, metrics = self.predict(model, X_test, y_test)
            results.append(
                Result(
                    trained_model=model,
                    metrics=metrics,
                    y_pred=y_pred,
                    y_true=y_test,
                )
            )
        return results

    def train():
        pass

    def predict():
        pass


class BinaryClassificationExperiment(Experiment):
    epochs = 50

    def __init__(self) -> None:
        super().__init__()

    def train(
        self,
        X_train,
        y_train,
        model_architecture,
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        load_trained_model=False,
        validation_split=0.1,
        # TODO: yitong do something with model_save_name but dont let experiment over write it...
        model_save_name=None,
    ):
        def scheduler(epoch, lr):
            if epoch < 5:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)
        es_scheduler = keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", verbose=1, patience=10 ^ 4
        )
        if model_save_name is not None:
            mc_scheduler = keras.callbacks.ModelCheckpoint(
                model_save_name, monitor="val_loss", mode="min"
            )
        else: 
            mc_scheduler= None

        if validation_split > 0:
            (X_train, X_val, y_train, y_val) = train_test_split(
                X_train,
                y_train,
                test_size=validation_split,
                shuffle=True,
                random_state=5,
            )
        else:
            (X_val, y_val) = (np.array([]), np.array([]))

        if load_trained_model and model_save_name is not None:
            model = keras.models.load_model(model_save_name)
        else:
            model = model_architecture(optimizer)
            model.fit(
                x=X_train,
                y=y_train,
                validation_data=(X_val, y_val),
                batch_size=128,
                epochs=self.epochs,
                verbose="auto",
                initial_epoch=0,
                # class_weight={1: 0.5, 0: 0.5},
                validation_freq=1,
                use_multiprocessing=False,
                callbacks=[lr_scheduler, es_scheduler]
                + ([mc_scheduler] if mc_scheduler is not None else []),
            )
        return model

    def predict(self, model, X_test, y_true):
        # TODO(Yitong) We probably want to use evaluation.classifcation_evaluation here...
        return [], {}
