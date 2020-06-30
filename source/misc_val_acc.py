from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import sys
from pathlib import Path, PurePath
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as iointe
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image  # Pillow Pil
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models, transforms
from torch.optim import lr_scheduler
import sys
from tabulate import tabulate
import copy
BATCH_SIZE = 3

"""
Example of calculating accuracy

"""

a = torch.tensor([[0, 0.5, 0.5, 0, 0], [1.0, 0, 0, 0]])
b = torch.tensor([[0, 0.6, 0.4, 0, 0], [0.5, 0, 0.5, 0]])


def calc_batch_acc(input_emotions, prediction_emotions):
    """
    [[0, 0.5, 0.5, 0, 0], [1.0, 0, 0, 0]] # a = input
    [[0, 0.6, 0.4, 0, 0], [0.5, 0, 0.5, 0]] # b = output

    0, -0.1, 0.1, 0     0.5, 0, -0.5, 0   # step 1 finding differences (a - b)

    0, 0.1, 0.1, 0, 0   0.5, 0, 0.5,  0   # step 2 abs of differences

    0.2, 0.5                            # step 3 summing errors

    2 - 0.2, 2 - 0.5                    # step 4 scaling errors to 2
    1.8, 1.5                            # step 5 (0 = very bad, ..., 1 = ok, ..., 2 = perfect)

    0.9, 0.75                           # step 6 scaling to [0, 1] by diving with /2

    90% correct prediction for first vector
    75% correct prediction for second vector
    """
    print("input:", input_emotions)
    print("prediction:", prediction_emotions)

    div = torch.abs(input_emotions - prediction_emotions)
    sums = torch.sum(div, dim=1)
    sums2 = (2 - sums)/2
    res = torch.mean(sums2)

    print("DIV"+str(div), "Sums "+str(sums), "Sums2"+str(sums2), "Res "+str(res),  sep='\n')


calc_batch_acc(a, b)
