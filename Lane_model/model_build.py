import time
from glob import glob  # Finds all path names matching specified pattern

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from batch_data_generator import BatchDataGenerator
from constants import (
    BATCH_SIZE,
    LR,
    NB_EPOCHS,
    RANDOM_STATE,
    TEST_SIZE,
    TRAIN_DATA_DIR,
    TRAIN_LABELS_DIR,
    VERBOSITY,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

from models import ltc_model

# Get the HDF5 sunny camera and log files
cam_files = sorted(glob(TRAIN_DATA_DIR))
log_files = sorted(glob(TRAIN_LABELS_DIR))

X_train, X_val, y_train, y_val = train_test_split(
    cam_files, log_files, test_size=TEST_SIZE, random_state=RANDOM_STATE
)


# Instantiate the train, validation and test data generators
""" train_gen = DataGenerator(X_train, y_train)
val_gen = DataGenerator(X_val, y_val) """

train_gen = BatchDataGenerator(X_train, y_train, batch_size=BATCH_SIZE, normalize_labels=True)
val_gen = BatchDataGenerator(X_val, y_val, batch_size=BATCH_SIZE, normalize_labels=True)

# Store architecture function references and model names in a dictionary.
# This dictionary is utilised to get the necessary model from the parsed command line arguments.
arch_dict = {"ltc": [ltc_model, "ltc", "CNN_LTC"]}

model, model_name, model_name_plot = arch_dict["ltc"]

# Initialise the optimiser
optimizer = Adam(learning_rate=LR)

# Get the Keras sequential/functional model by using its reference
model = model()

model.compile(
    loss="mean_squared_error", optimizer=optimizer, metrics=["mean_squared_error"]
)


cps_path = f"models/{model_name}_model" + "-{val_loss:03f}.h5"

# Create a Keras 'ModelCheckpoint' callback to save the best model
checkpoint = ModelCheckpoint(
    cps_path,
    monitor="val_mean_squared_error",
    verbose=VERBOSITY,
    save_best_only=True,
    mode="auto",
)
NAME = f"{model_name}_model_{int(time.time())}"
tensorboard = TensorBoard(
    log_dir="logs/{}".format(NAME),
    write_graph=True,
    update_freq="epoch",
    histogram_freq=1,
)

# Start training the model
history = model.fit(
    train_gen,
    epochs=NB_EPOCHS,
    verbose=VERBOSITY,
    validation_data=val_gen,
    callbacks=[checkpoint, tensorboard]
)

# Plot the training and validation losses
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title(f"{model_name_plot} Model Loss (learning rate: {LR})")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train_loss", "val_loss"], loc="upper left")
plt.show()
