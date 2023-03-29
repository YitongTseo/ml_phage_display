import tensorflow as tf
from tensorflow.keras import layers, activations
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Recall, Precision, mean_squared_error


def p_value_and_fc_rmse(y_true, y_pred):
    return (p_value_rmse(y_true, y_pred) + fold_rmse(y_true, y_pred)) / 2

def p_value_rmse(y_true, y_pred):
    y_true_pvalue = y_true[:, 0]
    y_pred_pvalue = y_pred[:, 0]
    return tf.math.sqrt(mean_squared_error(y_true_pvalue, y_pred_pvalue))


def fold_rmse(y_true, y_pred):
    y_true_fold = y_true[:, 1]
    y_pred_fold = y_pred[:, 1]
    return tf.math.sqrt(mean_squared_error(y_true_fold, y_pred_fold))


def er_rmse(y_true, y_pred):
    y_true_fold = y_true[:, 2]
    y_pred_fold = y_pred[:, 2]
    return tf.math.sqrt(mean_squared_error(y_true_fold, y_pred_fold))


def multi_channel_mse(y_true, y_pred):
    # Be sure to normalize otherwise scales will not be similar
    squared_difference = tf.math.abs(y_true - y_pred) ** 2 
    return tf.reduce_mean(squared_difference, axis=-1)


def ThreeChannelRegressionRNN_gelu(optimizer, width=64, depth=6, dropout=0.1, loss=multi_channel_mse):
    # create model
    model = Sequential()
    assert width >= 16, "Can't have parameters go too low (>= 5 please)"
    assert depth >= 6, "Can't have depth go too low (>= 5 please)"

    model.add(layers.Dense(width, kernel_initializer="he_uniform"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.gelu))
    model.add(layers.Dropout(dropout))

    model.add(layers.Bidirectional(layers.LSTM(width)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.gelu))
    model.add(layers.Dropout(dropout))

    for depth_idx in range(depth - 6):
        model.add(layers.Dense(width, kernel_initializer="he_uniform"))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(activations.gelu))
        model.add(layers.Dropout(dropout))
        
    model.add(layers.Dense(int(width / 2), kernel_initializer="he_uniform"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.gelu))
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(int(width / 4), kernel_initializer="he_uniform"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.gelu))
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(int(width / 8), kernel_initializer="he_uniform"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.gelu))
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(3, kernel_initializer="normal", use_bias=True))
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[multi_channel_mse, fold_rmse, p_value_rmse, er_rmse],
        run_eagerly=True,
    )

    return model