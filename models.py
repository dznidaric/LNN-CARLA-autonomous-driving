from keras.layers import (
    RNN,
    Add,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    MaxPool2D,
    Reshape,
)
from keras.models import Model, Sequential
from keras.utils import plot_model
from kerasncp.tf import LTCCell
from kerasncp.wirings import NCP

from constants_1 import INPUT_SHAPE


def cnn_ncp_model():
    """
    Define the Keras sequential CNN-NCP hybrid architecture.

    A CNN feature extractor is stacked with an NCP RNN temporal modelling structure.

    :return: the CNN-NCP model
    """

    # Set the NCP wiring
    wiring = NCP(
        inter_neurons=12,  # Number of inter neurons
        command_neurons=8,  # Number of command neurons
        motor_neurons=1,  # Number of motor neurons
        sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
        inter_fanout=4,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=4,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=6,  # How many incoming synapses has each motor neuron
    )

    # Create the the NCP cell based on the LTC neuron.
    ncp_cell = LTCCell(wiring)

    # Build the sequential hybrid model
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 2, activation='relu'))
    model.add(Conv2D(36, 5, 2, activation='relu'))
    model.add(Conv2D(48, 5, 2, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Reshape((1, -1)))
    model.add(RNN(ncp_cell, unroll=True))
    model.summary()
    # plot_model(model, show_shapes=True)
    return model


def cnn_dncp_model():
    """
    Define the blended functional CNN-DNCP v2 model.

    Two NCP wiring settings are adopted representing the left and right sides of the human brain.
    This model is, therefore, called DNCP which is short for Dual NCP.

    :return: the brain-inspired CNN-DNCP model
    """

    # Set the NCP wiring configurations
    left_wiring = NCP(
        inter_neurons=3,  # Number of inter neurons
        command_neurons=5,  # Number of command neurons
        motor_neurons=1,  # (1 output) Number of motor neurons
        sensory_fanout=2,  # How many outgoing synapses has each sensory neuron
        inter_fanout=2,  # (average=15/8) How many outgoing synapses has each inter neuron
        recurrent_command_synapses=1,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=2,  # (12/5) How many incoming synapses has each motor neuron
    )

    right_wiring = NCP(
        inter_neurons=4,  # Number of inter neurons
        command_neurons=6,  # Number of command neurons
        motor_neurons=1,  # Number of motor neurons
        sensory_fanout=2,  # (18/8) How many outgoing synapses has each sensory neuron
        inter_fanout=2,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=2,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=2,  # How many incoming synapses has each motor neuron
    )

    # Create the left and right brain-inspired NCP cells.
    # These are based on the LTC neuron.
    ncp_cell_left = LTCCell(left_wiring)
    ncp_cell_right = LTCCell(right_wiring)

    # Build the functional brain-inspired hybrid model
    input_lyr = Input(shape=INPUT_SHAPE)
    lambda_lyr = Lambda(lambda x: x / 127.5 - 1.0)(input_lyr)
    conv_1 = Conv2D(24, 5, 2, activation='relu')(lambda_lyr)
    conv_2 = Conv2D(36, 5, 2, activation='relu')(conv_1)
    conv_3 = Conv2D(48, 5, 2, activation='relu')(conv_2)
    conv_4 = Conv2D(64, 3, activation='relu')(conv_3)
    conv_5 = Conv2D(64, 3, activation='relu')(conv_4)
    dropout_lyr = Dropout(0.5)(conv_5)
    flatten_lyr = Flatten()(dropout_lyr)
    dense_1 = Dense(100, activation='relu')(flatten_lyr)
    reshape_lyr = Reshape((1, -1))(dense_1)
    ncp_left = RNN(ncp_cell_left, unroll=True)(reshape_lyr)
    ncp_right = RNN(ncp_cell_right, unroll=True)(reshape_lyr)
    ncps_added = Add()([ncp_left, ncp_right])
    output = Lambda(lambda x: x * 0.5)(ncps_added)
    model = Model(inputs=input_lyr, outputs=output)
    model.summary()
    # plot_model(model, show_shapes=True)
    return model
