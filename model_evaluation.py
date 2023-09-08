"""
Python program employed in evaluating the various
developed architecture pretrained models. Along with this
by running the script a prediction plot is created.
"""

# Importing Python libraries
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Disable debugging logs

import argparse
import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from kerasncp.tf import LTCCell
from sklearn.utils import shuffle
from tensorflow.python.keras.models import load_model

from data_gen import DataGenerator

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Model evaluation program.")

# Fill the ArgumentParser object with information about program arguments.
parser.add_argument(
    "model",
    type=str,
    metavar="model",
    choices=["ncp"],
    help="model type",
)

parser.add_argument(
    "gen",
    type=str,
    metavar="gen",
    choices=["cloudy", "night"],
    help="evaluation data generator type",
)

parser.add_argument(
    "-unseeded", action="store_true", help="leave the random script values unseeded"
)

parser.add_argument(
    "-path", action="store", help="enable the user to define a pretrained model path"
)

# Parse the arguments
args = parser.parse_args()


def eval_model(model, mdl_name, gen):
    """
    Evaluate model and print the test loss

    :param tf.keras.Model model:
    :param string mdl_name: name of the model
    :param DataGenerator gen: DataGenerator instance (test data generator in this case)
    :return: None
    """

    print(f"\n{mdl_name} model evaluation:")
    res = model.evaluate(gen)
    print(f"Test loss (MSE): {res}")


def plot_angles(mdl_name, actual, predicted):
    """
    Plot the actual vs predicted steering angle values.

    :param mdl_name: name of the model
    :param actual: actual steering angle values data
    :param predicted: predicted steering angle values data
    :return: None
    """

    plt.figure(figsize=(11, 6))
    plt.plot(actual, linestyle="solid", color="r")
    plt.plot(predicted, linestyle="dashed", color="b")
    plt.legend(["Actual", "Predicted"], loc="best", prop={"size": 14})
    plt.title(
        f"{mdl_name} Actual vs Predicted Steering Angle Values",
        weight="bold",
        fontsize=16,
    )
    plt.ylabel("Angle (scaled)", weight="bold", fontsize=14)
    plt.xlabel("Batch Sample no.", weight="bold", fontsize=14)
    plt.xticks(weight="bold", fontsize=12, rotation=45)
    plt.yticks(weight="bold", fontsize=12)
    plt.grid(color="y", linewidth="0.5")
    plt.show()


def eval_predict(mdl_path, mdl_name, gen, custom_lyr=LTCCell):
    """
    Evaluate the model and plot its angle predictions agains the actual ones.

    :param mdl_path: path to the model
    :param mdl_name: name of the model
    :param gen: evaluation data generator
    :param custom_lyr: custom layer (hybrid models)
    :return: None
    """
    # Load and evaluate the CNN checkpoint model
    model = load_model(mdl_path, custom_objects={'LTCCell': custom_lyr})
    model.summary()
    eval_model(model, mdl_name, gen)

    # Predict angles for the first 64 frames
    # Predict steering angle output values using a single batch from the test data generator
    y_pred = model(gen[0][0])  # Predicted steering angle output values
    y_actual = gen[0][1]  # Actual (expected) steering angle output values
    plot_angles(model_name, y_actual, y_pred)


# Fetch the batched dataset frame and angle NumPy filenames
# Here we adopt the cloudy dataset recording files to evaluate
# the checkpoint models (best loss).
eval_gen_type = args.gen
cam_files = sorted(glob(f"./dataset/64_batched_{eval_gen_type}/frames/*.npy"))
log_files = sorted(glob(f"./dataset/64_batched_{eval_gen_type}/angles/*.npy"))

# Shuffle the frame and angle filenames
cam_files, log_files = shuffle(cam_files, log_files, random_state=42)

# Initialize the evaluation data generator
eval_gen = DataGenerator(cam_files, log_files)

# Set the dictionary values to the seeded or unseeded pretrained version of the models.
# Consider the '-unseeded' command line argument.
arch_dict = {
    "ncp": ["models/cnn_ncp_model-0.002525.h5", "CNN-NCP"]
}

try:
    # Fetch the model and the model and plot names
    model_path, model_name = arch_dict[args.model]

    # Run the model evaluator and prediction plot function
    eval_predict(model_path, model_name, eval_gen)
except OSError:
    print("Incorrect model path. Please try again!")
