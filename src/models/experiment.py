import pathlib
import sys

HOME_DIRECTORY = pathlib.Path().absolute()
sys.path.append(str(HOME_DIRECTORY) + "/src")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from typing import Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from models.rnn import two_channel_mse


@dataclass
class Result:
    metrics: dict[str, Any] = None
    trained_model: Optional[Any] = None
    y_pred: Optional[np.array] = None
    y_true: Optional[np.array] = None


class BinaryClassificationExperiment:
    def train(
        self,
        X_train,
        y_train,
        model_architecture,
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        load_trained_model=None
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
            validation_split=0.0,
            initial_epoch=0,
            # class_weight={1: 0.5, 0: 0.5},
            validation_freq=1,
            use_multiprocessing=False,
        )
        return model

    def predict(self, model, X_test, y_true):
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
        return (
            y_pred,
            {
                "accuracy": acc,
                "Precision": pre,
                "Recall": rec,
            },
        )

    def run_adhoc_experiment(
        self, X, y, model_architecture, test_train_split=0.2, load_trained_model=False
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


class RegressionExperiment(BinaryClassificationExperiment):
    def predict(self, model, X_test, y_true):
        y_pred = model(X_test)
        return (
            y_pred,
            {
                "mse": np.mean(two_channel_mse(y_true, y_pred)),
            },
        )

    def train(
        self,
        X_train,
        y_train,
        model_architecture,
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        load_trained_model=False,
        validation_split=0.1,
    ):
        def scheduler(epoch, lr):
            if epoch < 5:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)
        es_scheduler = keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", verbose=1, patience=3
        )
        mc_scheduler = keras.callbacks.ModelCheckpoint(
            "best_model.h5", monitor="val_loss", mode="min"
        )

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

        if load_trained_model:
            model = keras.models.load_model(
                "best_model.h5", custom_objects={"two_channel_mse": two_channel_mse}
            )
        else:
            model = model_architecture(optimizer)
            model.fit(
                x=X_train,
                y=y_train,
                validation_data=(X_val, y_val),
                batch_size=128,
                epochs=16,
                verbose="auto",
                initial_epoch=0,
                validation_freq=1,
                use_multiprocessing=False,
                callbacks=[lr_scheduler, es_scheduler, mc_scheduler],
            )

        return model
    
    
    
class SingleRegressionExperiment(BinaryClassificationExperiment):
    def predict(self, model, X_test, y_true):
        y_pred = model(X_test)
        return (
            y_pred,
                {
                    "mse": np.mean(rmse(y_true, y_pred)),
            },
        )
    

    def train(
        self,
        X_train,
        y_train,
        model_architecture,
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        load_trained_model=False,
        validation_split=0.1,
    ):
        def scheduler(epoch, lr):
            if epoch < 5:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)
        es_scheduler = keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", verbose=1, patience=3
        )
        mc_scheduler = keras.callbacks.ModelCheckpoint(
            "best_model.h5", monitor="val_loss", mode="min"
        )

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

        if load_trained_model:
            model = keras.models.load_model(
                "best_model.h5", custom_objects={"rmse": rmse}
            )
        else:
            model = model_architecture(optimizer)
            model.fit(
                x=X_train,
                y=y_train,
                validation_data=(X_val, y_val),
                batch_size=128,
                epochs=16,
                verbose="auto",
                initial_epoch=0,
                validation_freq=1,
                use_multiprocessing=False,
                callbacks=[lr_scheduler, es_scheduler, mc_scheduler],
            )

        return model

