import numpy as np
import tensorflow as tf

from config import GlobalConfig
from data import CARLA_Data


def main():
    config = GlobalConfig(setting="02_05_withheld")

    train_set = CARLA_Data(root=config.train_data, config=config)
    val_set = CARLA_Data(root=config.val_data, config=config)

    train_set.labels
    train_set.images[0].shape

    """  train_set = train_set.shuffle(1000).batch(32)
    val_set   = val_set.shuffle(1000).batch(32) """


def model():
    # Define the input shapes
    camera_input_shape = (124, 124, 3)
    lidar_input_shape = (1000, 3)

    # Define the camera encoder
    """  camera_encoder = tf.keras.models.Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=camera_input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
        ]
    )

    # Define the merge layer
    merge_layer = Concatenate()

    # Define the decoder
    decoder = tf.keras.models.Sequential(
        [
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    # Define the model
    model = Model(inputs=[camera_input_shape, lidar_input_shape], outputs=decoder)

    # Compile the model
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train the model
    data = ...  # Load your training data here
    labels = ...  # Load your training labels here
    model.fit(data, labels, epochs=100, batch_size=32) """


if __name__ == "__main__":
    main()
