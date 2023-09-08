TRAIN_DATA_DIR = "./64_batched_data_town05/images/*.npy"
TRAIN_LABELS_DIR = "./64_batched_data_town05/steering/*.npy"

TEST_SIZE = 0.2
RANDOM_STATE = 42

B_SIZE = 64
NB_EPOCHS = 10
LR = 5e-06  # Since trains the model to the best val_loss in 10 epochs (model starts overfitting)
VERBOSITY = 1

INPUT_SHAPE = (128, 640, 3)  # Image input shape
