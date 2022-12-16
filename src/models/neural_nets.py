# TODO: we probably want to make this HOME_DIRECTORY specification a bit cleaner
import pathlib

HOME_DIRECTORY = pathlib.Path().absolute().parent

import operator
import sys
import os
import pdb

sys.path.append(os.path.join(HOME_DIRECTORY, "DELPHI/Feature_Computation"))
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import Pro2Vec_1D.compute as Pro2Vec
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import umap
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from typing import Optional, Any
from dataclasses import dataclass


@dataclass
class Result:
    metrics: Optional[dict[str, Any]] = None
    trained_models: Optional[Any] = None
    y_pred: Optional[np.array] = None
    y_true: Optional[np.array] = None


def Transformer(optimizer):
    class TransformerBlock(layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
            super(TransformerBlock, self).__init__()
            self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.ffn = keras.Sequential(
                [
                    layers.Dense(ff_dim, activation="relu"),
                    layers.Dense(embed_dim),
                ]
            )
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = layers.Dropout(rate)
            self.dropout2 = layers.Dropout(rate)

        def call(self, inputs, training):
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)

    # create model
    model = Sequential()
    model.add(layers.Dense(64, activation="tanh"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(TransformerBlock(16, 5, 16, rate=0.1))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(16, activation="tanh"))
    model.add(layers.Dense(1, activation="sigmoid"))
    # Compile model
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy", Recall(), Precision()],
    )
    return model


def RNN(optimizer):
    # create model
    model = Sequential()
    model.add(layers.Dense(16, activation="tanh"))
    model.add(layers.Bidirectional(layers.LSTM(16)))
    model.add(layers.Dense(16, activation="tanh"))
    model.add(layers.Dense(8, activation="relu"))
    model.add(layers.Dense(4, activation="tanh"))
    model.add(layers.Dense(1, activation="sigmoid"))
    # Compile model
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy", Recall(), Precision()],
    )
    return model

def train(
    X,
    y,
    model_architecture=RNN,
    n_splits=5
):
    def scheduler(epoch, lr):
        if epoch < 5:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
    callback = keras.callbacks.LearningRateScheduler(scheduler)

    trained_models = []
    if n_splits > 1:
        kf = KFold(n_splits=5)
    accs, pres, recs = [], [], []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        optimizer=keras.optimizers.Adam(learning_rate=0.001)
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
        y_pred, acc, pre, rec = predict(model, X_test, y_test)
        accs.append(acc)
        pres.append(pre)
        recs.append(rec)
        trained_models.append(model)
    return Result(
        trained_models=trained_models,
        metrics={
            "Cross_val_accuracies": accs,
            "Cross_val_precisions": pres,
            "Cross_val_recalls": recs,
        },
        # TODO: batch together y_pred
        y_pred=None,
    )


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
