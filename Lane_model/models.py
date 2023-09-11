
from constants import PROCESSED_IMG_SHAPE
from keras.layers import RNN, Conv2D, Dense, Dropout, Flatten, InputLayer, Reshape
from keras.models import Sequential
from kerasncp.tf import LTCCell
from kerasncp.wirings import NCP


def ltc_model():

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
    model.add(InputLayer(input_shape=PROCESSED_IMG_SHAPE))
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