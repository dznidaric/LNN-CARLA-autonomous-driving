import os

DATA_DIR = r"/data"

class GlobalConfig:

    img_resolution = (160, 704)  # (height, width)
    img_width = 320

    # Training parameters
    augment = True
    inv_augment_prob = 0.1  # Probablity that data augmentation is applied is 1.0
    aug_max_rotation = 20  # degree

    def __init__(self, setting="all"):

        if setting == "all":  # All towns used for training no validation data
            self.train_towns = os.listdir(DATA_DIR)
            self.val_towns = [self.train_towns[0]]
            self.train_data, self.val_data = [], []
            for town in self.train_towns:
                root_files = os.listdir(
                    os.path.join(DATA_DIR, town)
                )  # Town folders
                for file in root_files:
                    if not os.path.isfile(os.path.join(DATA_DIR, file)):
                        self.train_data.append(os.path.join(DATA_DIR, town, file))
            for town in self.val_towns:
                root_files = os.listdir(os.path.join(DATA_DIR, town))
                for file in root_files:
                    if not os.path.isfile(os.path.join(DATA_DIR, file)):
                        self.val_data.append(os.path.join(DATA_DIR, town, file))

        elif setting == "02_05_withheld":  # Town02 and 05 withheld during training
            print("Skip Town02 and Town05")
            self.train_towns = os.listdir(DATA_DIR)  # Scenario Folders
            self.val_towns = (
                self.train_towns
            )  # Town 02 and 05 get selected automatically below
            self.train_data, self.val_data = [], []
            for town in self.train_towns:
                root_files = os.listdir(
                    os.path.join(DATA_DIR, town)
                )  # Town folders
                for file in root_files:
                    if (file.find("Town02") != -1) or (
                        file.find("Town05") != -1
                    ):  # We don't train on 05 and 02 to reserve them as test towns
                        continue
                    if not os.path.isfile(os.path.join(DATA_DIR, file)):
                        print("Train Folder: ", file)
                        self.train_data.append(os.path.join(DATA_DIR, town, file))
            for town in self.val_towns:
                root_files = os.listdir(os.path.join(DATA_DIR, town))
                for file in root_files:
                    if (file.find("Town02") == -1) and (
                        file.find("Town05") == -1
                    ):  # Only use Town 02 and 05 for validation
                        continue
                    if not os.path.isfile(os.path.join(DATA_DIR, file)):
                        print("Val Folder: ", file)
                        self.val_data.append(os.path.join(DATA_DIR, town, file))
        elif setting == "eval":  # No training data needed during evaluation.
            pass
        else:
            print("Error: Selected setting: ", setting, " does not exist.")
