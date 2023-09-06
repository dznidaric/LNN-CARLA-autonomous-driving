import time

import kerasncp as kncp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.layers import (
    RNN,
    Activation,
    Conv2D,
    Conv3D,
    Dense,
    Dropout,
    Flatten,
    Input,
    InputLayer,
    MaxPool2D,
    MaxPooling3D,
    Reshape,
    TimeDistributed,
)
from keras.models import Sequential
from keras.utils import plot_model
from kerasncp.tf import LTCCell

# from kerasncp.ltc_cell import LTCCell
from kerasncp.wirings import NCP
from ncps import wirings
from ncps.tf import LTC
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from config import GlobalConfig
from data_3D import CARLA_Data

NAME = "LTC-Carla-{}".format(int(time.time()))
tensorboard = TensorBoard(
    log_dir="logs/{}".format(NAME),
    write_graph=True,
    update_freq="batch",
    histogram_freq=1,
)
checkpoint_callback = ModelCheckpoint(
    filepath="models/LTC_CNN3D_model-{val_accuracy:.4f}.hdf5",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=False,
    mode="max",
    verbose=1,
)


def main():
    config = GlobalConfig(setting="02_05_withheld")

    model = LTC_CNN3D_model()

    histories = []

    for epoch in range(config.epochs):
        train_set = CARLA_Data(
            root=config.train_data, current_epoch=epoch, config=config
        ).create_dataset()
        val_set = CARLA_Data(
            root=config.val_data, current_epoch=epoch, config=config
        ).create_dataset()

        history = model.fit(
            train_set,
            epochs=1,
            validation_data=val_set,
            callbacks=[tensorboard, checkpoint_callback],
        )
        histories.append(history)

    plot_model_performance(histories)


def LTC_CNN3D_model():
    wiring = wirings.AutoNCP(12, 5)

    model = Sequential()
    model.add(
        Conv3D(
            32,
            (3, 3, 3),
            activation="relu",
            input_shape=(10, 160, 704, 3),
            padding="same",
        )
    )
    model.add(Conv3D(32, (3, 3, 3), activation="relu", padding="same"))
    model.add(
        MaxPooling3D(pool_size=(3, 3, 3), padding="same", data_format="channels_first")
    )
    model.add(Dropout(0.25))

    model.add(Conv3D(64, (3, 3, 3), activation="relu", padding="same"))
    model.add(Conv3D(64, (3, 3, 3), activation="relu", padding="same"))
    model.add(
        MaxPooling3D(pool_size=(3, 3, 3), padding="same", data_format="channels_first")
    )
    model.add(Dropout(0.25))

    model.add(TimeDistributed(Flatten()))
    model.add(Dense(100, activation="relu"))
    model.add(LTC(wiring, return_sequences=True))

    model.compile(
        optimizer=Adam(0.01),
        loss=MeanSquaredError(),
        metrics=["mean_squared_error", "accuracy"],
    )
    return model


def plot_model_performance(histories):
    plt.figure(figsize=(10, 6))

    for history in histories:
        df = pd.DataFrame(history.history)
        df["epoch"] = range(1, len(df) + 1)  # Add an 'epoch' column
        plt.plot("epoch", "loss", data=df, label="Loss")  # Plot loss
        plt.plot("epoch", "accuracy", data=df, label="Accuracy")  # Plot accuracy

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Model Performance by Epoch")
    plt.legend()
    plt.show()


def LTC_model():
    wiring = NCP(
        inter_neurons=12,  # Number of inter neurons
        command_neurons=8,  # Number of command neurons
        motor_neurons=4,  # Number of motor neurons
        sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
        inter_fanout=4,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=4,  # How many recurrent synapses are in the command neuron layer
        motor_fanin=6,  # How many incoming synapses has each motor neuron
    )
    # Create the the NCP cell based on the LTC neuron.
    ncp_cell = LTCCell(wiring)

    # Build the sequential hybrid model
    model = Sequential(
        [
            Conv2D(
                24, 5, 2, padding="same", activation="relu", input_shape=(160, 704, 3)
            ),
            Conv2D(36, 5, 2, activation="relu"),
            Conv2D(48, 5, 2, activation="relu"),
            Conv2D(64, 3, activation="relu"),
            Conv2D(64, 3, activation="relu"),
            Dropout(0.5),
            Flatten(),
            Dense(100, activation="relu"),
            Reshape((1, -1)),
            RNN(ncp_cell, unroll=True),
        ]
    )

    model.summary()
    # plot_model(model, show_shapes=True)
    # return model


def LTC_time_model():
    ncp_wiring = kncp.wirings.NCP(
        inter_neurons=20,  # Number of inter neurons
        command_neurons=10,  # Number of command neurons
        motor_neurons=5,  # Number of motor neurons
        sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
        inter_fanout=5,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=6,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=4,  # How many incoming synapses has each motor neuron
    )
    ncp_cell = LTCCell(
        ncp_wiring,
        initialization_ranges={
            # Overwrite some of the initialization ranges
            "w": (0.2, 2.0),
        },
    )
    model = Sequential(
    [
        InputLayer(input_shape=(None, 160, 704, 3)),
        TimeDistributed(
            Conv2D(32, (5, 5), activation="relu")
        ),
        TimeDistributed(MaxPool2D()),
        TimeDistributed(
            Conv2D(64, (5, 5), activation="relu")
        ),
        TimeDistributed(MaxPool2D()),
        TimeDistributed(Flatten()),
        TimeDistributed(Dense(32, activation="relu")),
        RNN(ncp_cell, return_sequences=True),
        TimeDistributed(Activation("softmax")),
    ])
    model.compile(
        optimizer=Adam(0.01),
        loss='mse',
    )
    return model


if __name__ == "__main__":
    main()
