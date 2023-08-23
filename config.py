""" 
@article{Chitta2022PAMI,
  author = {Chitta, Kashyap and
            Prakash, Aditya and
            Jaeger, Bernhard and
            Yu, Zehao and
            Renz, Katrin and
            Geiger, Andreas},
  title = {TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving},
  journal = {Pattern Analysis and Machine Intelligence (PAMI)},
  year = {2022},
} 
@inproceedings{Prakash2021CVPR,
  author = {Prakash, Aditya and
            Chitta, Kashyap and
            Geiger, Andreas},
  title = {Multi-Modal Fusion Transformer for End-to-End Autonomous Driving},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2021}
} """

""" github repo: https://github.com/autonomousvision/transfuser#evaluation """

import os

DATA_DIR = r"./data"


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
                root_files = os.listdir(os.path.join(DATA_DIR, town))  # Town folders
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
                root_files = os.listdir(os.path.join(DATA_DIR, town))  # Town folders
                for file in root_files:
                    if (file.find("Town02") != -1) or (file.find("Town05") != -1):
                        if not os.path.isfile(os.path.join(DATA_DIR, file)):
                            print("Val Folder: ", file)
                            self.val_data.append(os.path.join(DATA_DIR, town, file))
                            continue
                    if not os.path.isfile(os.path.join(DATA_DIR, file)):
                        print("Train Folder: ", file)
                        self.train_data.append(os.path.join(DATA_DIR, town, file))
        elif setting == "eval":  # No training data needed during evaluation.
            pass
        else:
            print("Error: Selected setting: ", setting, " does not exist.")
