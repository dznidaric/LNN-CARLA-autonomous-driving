TRAIN_DATA_DIR = "./64_batched_data_town05/images/*.npy"
TRAIN_LABELS_DIR = "./64_batched_data_town05/labels/*.npy"

TEST_SIZE = 0.2
RANDOM_STATE = 42

BATCH_SIZE = 64
NB_EPOCHS = 100
LR = 0.01  # Since trains the model to the best val_loss in 10 epochs (model starts overfitting)
VERBOSITY = 1

INPUT_SHAPE = (360, 640)  # Image input shape

PROCESSED_IMG_SHAPE = (144, 320, 1)  # Image shape after processing
