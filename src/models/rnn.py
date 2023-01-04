import tensorflow as tf
from tensorflow.keras import layers, activations
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Recall, Precision
import pdb

def rmse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))
    
def p_value_rmse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred)[:, 0])


def fold_rmse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred)[:, 1])


def two_channel_mse(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)


def RegressionRNN(optimizer):
    # create model
    model = Sequential()
    model.add(layers.Dense(16))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.tanh))
    model.add(layers.Dropout(0.01))

    model.add(layers.Bidirectional(layers.LSTM(16)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.tanh))
    model.add(layers.Dropout(0.01))

    model.add(layers.Dense(16))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.tanh))
    model.add(layers.Dropout(0.01))

    model.add(layers.Dense(8))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.tanh))
    model.add(layers.Dropout(0.01))

    model.add(layers.Dense(4, activation="tanh"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.tanh))
    model.add(layers.Dropout(0.01))

    model.add(
        layers.Dense(2, activation=None, kernel_initializer="normal", use_bias=True)
    )
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=two_channel_mse,
        metrics=[two_channel_mse, fold_rmse, p_value_rmse],
        run_eagerly=True,
    )
    return model

def SingleRegressionRNN(optimizer):
    # create model
    model = Sequential()
    model.add(layers.Dense(16))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.tanh))
    model.add(layers.Dropout(0.01))

    model.add(layers.Bidirectional(layers.LSTM(16)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.tanh))
    model.add(layers.Dropout(0.01))

    model.add(layers.Dense(16))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.tanh))
    model.add(layers.Dropout(0.01))

    model.add(layers.Dense(8))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.tanh))
    model.add(layers.Dropout(0.01))

    model.add(layers.Dense(4, activation="tanh"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.tanh))
    model.add(layers.Dropout(0.01))

    model.add(
        layers.Dense(1, activation=None, kernel_initializer="normal", use_bias=True)
    )
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=two_channel_mse,
        metrics=rmse,
        run_eagerly=True,
    )
    return model


def BinaryClassificationRNN(optimizer):
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
