import tensorflow as tf
from tensorflow.keras import layers, activations
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Recall, Precision, mean_squared_error
import pdb


def p_value_rmse(y_true, y_pred):
    y_true_pvalue = y_true[:, 0]
    y_pred_pvalue = y_pred[:, 0]
    return mean_squared_error(y_true_pvalue, y_pred_pvalue)


def fold_rmse(y_true, y_pred):
    y_true_fold = y_true[:, 1]
    y_pred_fold = y_pred[:, 1]
    return mean_squared_error(y_true_fold, y_pred_fold)


def two_channel_mse(y_true, y_pred):
    # scales are not similar
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)

def TwoChannelRegressionRNN(optimizer):
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



def SingleChannelRegressionRNN(optimizer):
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
        loss="mse",
        metrics=[mean_squared_error],
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
        # loss="mse",
        optimizer=optimizer,
        metrics=["accuracy", Recall(), Precision()],
    )
    return model

def Joint_BinaryClassificationRNN_gelu(optimizer):
    # create model
    model = Sequential()
    model.add(layers.Dense(16, activation="gelu"))
    model.add(layers.Bidirectional(layers.LSTM(16)))
    model.add(layers.Dense(16, activation="gelu"))
    model.add(layers.Dense(8, activation="gelu"))
    model.add(layers.Dense(4, activation="gelu"))
    model.add(layers.Dense(2, activation="sigmoid"))
    # Compile model
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy", Recall(), Precision()],
    )
    return model


def Joint_BinaryClassificationCNN_gelu(optimizer):
    # create model
    model = Sequential()
    model.add(layers.Conv1D(16, 3, activation='relu',  input_shape=(14,16)))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dense(16, activation="gelu"))
    model.add(layers.Dense(8, activation="gelu"))
    model.add(layers.Dense(4, activation="gelu"))
    model.add(layers.Dense(2, activation="sigmoid"))
    # Compile model
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy", Recall(), Precision()],
    )
    return model


