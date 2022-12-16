
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Recall, Precision

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
