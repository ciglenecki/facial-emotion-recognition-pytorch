"""
Configraution used for training
"""

import string
import random
from misc_input import *
import numpy as np
import torch
from misc_input import *
from config import *

torch.set_printoptions(sci_mode=False)


WEIGHTS_CK = np.array([2554.691, 443.665, 101.091, 384.644, 245.113, 584.815, 241.881, 577.1])
WEIGHTS_GOOGLE = np.array([328., 412., 186., 433., 281., 314., 412., 401.])


WEIGHTS = np.add(WEIGHTS_CK * CK_TRAIN_SPLIT, WEIGHTS_GOOGLE * GOOGLE_TRAIN_SPLIT)  # constuct weights based on SPLIT values
WEIGHTS = torch.from_numpy(np.amax(WEIGHTS)/np.array(WEIGHTS))  # normalize weights

DO_TRAIN_MODEL = bool_action("Train model")
if DO_TRAIN_MODEL:
    DATASET_DROP_RATE = number_action("Dataset drop rate")
    DO_USE_SCHEDULER = bool_action("Use LR scheduler")
    LEARNING_RATE = number_action("Learning rate")
    EPOCHS = int(number_action("Epochs"))
    OPTIMIZER = int(number_action("Sgd = 0, Adam = 1"))

AUG_DEGREE = 15  # random degree image
MODEL_SUFFIX = random_string()  # filenaming for temporary model saving
BATCH_SIZE = 16
NUM_WORKERS = 8
