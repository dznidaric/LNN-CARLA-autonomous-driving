import os
from glob import glob

import cv2
import numpy as np
from constants import TRAIN_DATA_DIR, TRAIN_LABELS_DIR

image_files = glob(TRAIN_DATA_DIR)
labels_files = glob(TRAIN_LABELS_DIR)


img = np.load(image_files[0])
lbl = np.load(labels_files[0])
print(img.shape)
for i in range(0, 64):
    print(lbl[i])
    cv2.imshow("", img[i])
    cv2.waitKey()