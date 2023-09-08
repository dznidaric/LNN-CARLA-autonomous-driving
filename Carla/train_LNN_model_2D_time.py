import time

import kerasncp as kncp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.layers import (
    LSTM,
    RNN,
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    InputLayer,
    MaxPool2D,
    TimeDistributed,
)
from keras.models import Sequential, load_model
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

NAME = "LTC-Carla-ts_{}".format(int(time.time()))
tensorboard = TensorBoard(
    log_dir="logs/{}".format(NAME),
    write_graph=True,
    update_freq="batch",
    histogram_freq=1,
)
checkpoint_callback = ModelCheckpoint(
    filepath="models/LTC_CNN2D_model-{val_accuracy:.4f}.h5",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=False,
    mode="max",
    verbose=1,
)

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
    model = Sequential()
    
    model.add(InputLayer(input_shape=(None, 160, 704, 3)))
    model.add(TimeDistributed(Conv2D(24, (5, 5), (2, 2), activation="relu",padding="same")))
    model.add(TimeDistributed(Conv2D(36, (5, 5), (2, 2), activation="relu")))
    model.add(TimeDistributed(Conv2D(48, (5, 5), (2, 2), activation="relu")))
    model.add(TimeDistributed(Conv2D(64, (5, 5), (2, 2), activation="relu")))
    model.add(TimeDistributed(Conv2D(64, (5, 5), (2, 2), activation="relu")))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(32, activation="relu")))
    model.add(RNN(ncp_cell, return_sequences=True))
    model.add(TimeDistributed(Activation("softmax")))
    model.compile(
        optimizer=Adam(0.01),
        loss="mse",
        metrics=["mean_squared_error", "accuracy"]
    )
    return model


config = GlobalConfig(setting="02_05_withheld")

model = load_model("models/LTC_CNN2D_model-0.8328.h5")

histories = []

epochs = 50

for epoch in range(epochs):
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