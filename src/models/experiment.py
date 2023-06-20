import pathlib
import sys

HOME_DIRECTORY = pathlib.Path().absolute()
sys.path.append(str(HOME_DIRECTORY) + "/src")

import os
import pdb
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from typing import Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from models.rnn import multi_channel_mse, p_value_rmse, fold_rmse, er_rmse
from tensorflow.keras.utils import custom_object_scope
from sklearn.preprocessing import StandardScaler


@dataclass
class Result:
    metrics: Any = None
    trained_model: Optional[Any] = None
    y_pred_rescaled: Optional[np.array] = None
    y_pred: Optional[np.array] = None
    y_test: Optional[np.array] = None
    X_test: Optional[np.array] = None
    yscaler: Optional[StandardScaler] = None


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
        other_datasets=[],
        normalize=False,
        batch_size=128,
        num_epochs=20,
    ):
        split_datasets = train_test_split(
            X,
            y,
            *other_datasets,
            test_size=test_train_split,
            shuffle=True,
            random_state=5,
        )
        X_train = split_datasets[0]
        X_test = split_datasets[1]
        y_train = split_datasets[2]
        y_test = split_datasets[3]

        if normalize:
            yscaler = StandardScaler()
            yscaler.fit(y_train)
            y_train = yscaler.transform(y_train)
            y_test = yscaler.transform(y_test)
        model = self.train(
            X_train,
            y_train,
            model_architecture,
            load_trained_model=load_trained_model,
            model_save_name=model_save_name,
            optimizer=optimizer,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )
        y_pred, metrics = self.predict(model, X_test, y_test)
        y_pred_rescaled = yscaler.inverse_transform(y_pred) if normalize else None
        return (
            split_datasets,
            Result(
                trained_model=model,
                metrics=metrics,
                y_pred=y_pred,
                y_test=y_test,
                yscaler=yscaler,
                y_pred_rescaled=y_pred_rescaled,
            ),
        )

    def run_cross_validation_experiment(
        self,
        X,
        y,
        model_architecture,
        optimizer,
        n_splits=5,
        load_trained_model=False,
        model_save_name=None,
        normalize=True,
        batch_size=128,
        num_epochs=5,
    ):
        kf = KFold(n_splits=n_splits)
        results = []
        for split_idx, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if normalize:
                yscaler = StandardScaler()
                yscaler.fit(y_train)
                y_train_scaled = yscaler.transform(y_train)
                y_test_saled = yscaler.transform(y_test)
            model = self.train(
                X_train,
                y_train_scaled,
                model_architecture,
                load_trained_model=load_trained_model,
                batch_size=batch_size,
                optimizer=optimizer(),
                num_epochs=num_epochs,
                model_save_name=model_save_name + str(split_idx),
            )
            y_pred, metrics = self.predict(model, X_test, y_test_saled)
            y_pred_rescaled = yscaler.inverse_transform(y_pred) if normalize else None
            results.append(
                Result(
                    trained_model=model,
                    metrics=metrics,
                    y_pred=y_pred,
                    y_pred_rescaled=y_pred_rescaled,
                    y_test=y_test,
                    X_test=X_test,
                    yscaler=yscaler,
                )
            )
        return results

    def train(
        self,
        X_train,
        y_train,
        model_architecture,
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        load_trained_model=False,
        batch_size=128,
        validation_split=0.0,
        model_save_name=None,
        num_epochs=20,
    ):
        if model_architecture.func.__name__ != "ThreeChannelRegressionRNN_gelu":
            X_train.reshape(X_train.shape[0], -1)
            model = model_architecture().fit(
                X_train.reshape(X_train.shape[0], -1), y_train
            )
            return model

        def scheduler(epoch, lr):
            if epoch < 5:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)
        es_scheduler = keras.callbacks.EarlyStopping(
            monitor="multi_channel_mse",
            mode="min",
            verbose=1,
            patience=5,  # restore_best_weights=True,
        )
        if model_save_name is not None:
            mc_scheduler = keras.callbacks.ModelCheckpoint(
                model_save_name, monitor="val_loss", mode="min", save_freq="epoch"
            )
        else:
            mc_scheduler = None

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
            assert ".h5" in model_save_name or os.path.isdir(
                model_save_name
            ), " either needs to be a .h5 filename or a directory to a .pb file"
            with custom_object_scope(
                {
                    "multi_channel_mse": multi_channel_mse,
                    "fold_rmse": fold_rmse,
                    "p_value_rmse": p_value_rmse,
                    "er_rmse": er_rmse,
                }
            ):
                model = keras.models.load_model(model_save_name)
        else:
            model = model_architecture(optimizer, output_size=y_train.shape[1])
            model.fit(
                x=X_train,
                y=y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=num_epochs,
                verbose="auto",
                initial_epoch=0,
                validation_freq=1,
                use_multiprocessing=False,
                callbacks=[lr_scheduler, es_scheduler]
                + ([mc_scheduler] if mc_scheduler is not None else []),
            )
        return model

    def predict(self, model, X_test, y_true):
        if isinstance(model, keras.models.Sequential):
            y_pred = model(X_test)
        else:
            y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
            y_pred = y_pred.reshape(y_true.shape)
        return y_pred, {
            "total_mse": multi_channel_mse(y_true, y_pred),
            # "p_value_rmse": p_value_rmse(y_true, y_pred),
            # "fold_rmse": fold_rmse(y_true, y_pred),
            # "er_rmse": er_rmse(y_true, y_pred),
        }


def pval_filter_ranking_lambda(x):
    return x[1] + x[2]
    # if x[0] > - np.log(0.01): # else try 2 #- np.log(0.01):
    #     return x[1] + x[2]
    # else:
    #     return 0
