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

a = torch.tensor([[0, 0.5, 0.5, 0, 0], [0.5, 0, 0.5, 0, 0], [0, 0, 0, 0.5, 0.5]])
b = torch.tensor([[0, 0.5, 0.5, 0, 0], [0.5, 0, 0, 0, 0], [0, 0, 0.5, 0.5, 0]])
c = torch.tensor([0, 0, 0, 0.5, 0.5])
d = torch.tensor([0, 0, 0.5, 0, 0.5])


def calc_batch_acc(outputs, emotions):
    """
    0.0, 0.5, 0.5, 0.0,   1.0, 0.0, 0.0 ,0.0
    0.5, 0.5, 0.0, 0.0,   0.5, 0.5, 0.0, 0.0

    -0.5, 0.0, 0.5, 0.0   0.5, -0.5, 0.0, 0.0

    0.5, 0.0, 0.5, 0.0    0.5, 0.5, 0.0, 0.0

    1 1
    1 1

    50% 50%
    """
    print("A:", outputs)
    print("B:", emotions)

    div = torch.abs(emotions - outputs)
    sums = torch.sum(div, dim=1)
    sums2 = (2 - sums)/2
    res = torch.mean(sums2)

    print("DIV"+str(div), "Sums "+str(sums), "Sums2"+str(sums2), "Res "+str(res),  sep='\n')


calc_batch_acc(a, b)
