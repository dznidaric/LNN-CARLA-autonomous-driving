"""
Python program employed in training the various developed architecture models.
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable debugging logs
import argparse
import time
from glob import glob  # Finds all path names matching specified pattern

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

from constants_1 import (
    LR,
    NB_EPOCHS,
    RANDOM_STATE,
    TEST_SIZE,
    TRAIN_DATA_DIR,
    TRAIN_LABELS_FL_DIR,
    VERBOSITY,
)
from data_gen import DataGenerator
from models import cnn_dncp_model, cnn_ncp_model

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Model training program.')

# Fill the ArgumentParser object with information about program arguments.
parser.add_argument(
    'model',
    type=str,
    metavar='model',
    choices=['ncp', 'dncp', 'dncp2', 'dncp3', 'dncp4'],
    default='ncp',
    help='model type'
)

# Parse the arguments
args = parser.parse_args()


# Get the HDF5 sunny camera and log files
cam_files = sorted(glob(TRAIN_DATA_DIR))
log_files = sorted(glob(TRAIN_LABELS_FL_DIR))

# Split the sunny comma.ai videos whose camera frame recordings have been shrunk to 80 x 120 (height x width).
# The dataset training to validation set ratio is 80/20.
X_train, X_val, y_train, y_val \
    = train_test_split(cam_files, log_files, test_size=TEST_SIZE, random_state=RANDOM_STATE)


# Instantiate the train, validation and test data generators
train_gen = DataGenerator(X_train, y_train)
val_gen = DataGenerator(X_val, y_val)

# Store architecture function references and model names in a dictionary.
# This dictionary is utilised to get the necessary model from the parsed command line arguments.
arch_dict = {'ncp': [cnn_ncp_model, 'cnn_ncp', 'CNN-NCP'],
             'dncp': [cnn_dncp_model, 'cnn_dncp', 'CNN-DNCP']
             }

model, model_name, model_name_plot = arch_dict[args.model]

optimizer = Adam(learning_rate=LR)

model = model()

model.compile(loss='mse',
              optimizer=optimizer,
              )


cps_path = f'models/{model_name}_model' + '-{val_loss:03f}.h5'

NAME = f"{model_name}_model_{int(time.time())}"
tensorboard = TensorBoard(
    log_dir="logs/{}".format(NAME),
    write_graph=True,
    update_freq="epoch",
    histogram_freq=1,
)

checkpoint = ModelCheckpoint(cps_path,
                             monitor='val_loss',
                             verbose=VERBOSITY,
                             save_best_only=True,
                             mode='auto')

history = model.fit(train_gen,
                    epochs=NB_EPOCHS,
                    verbose=VERBOSITY,
                    validation_data=val_gen,
                    callbacks=[checkpoint,tensorboard]
                    )

# Plot the training and validation losses
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(f'{model_name_plot} Model Loss (learning rate: {LR})')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'], loc='upper left')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title(f'{model_name_plot} Model Accuracy (learning rate: {LR})')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_accuracy', 'val_accuracy'], loc='upper left')
plt.show()

