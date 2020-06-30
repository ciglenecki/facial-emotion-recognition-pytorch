import os
import sys
from pathlib import Path, PurePath
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
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
import shutil
from os import listdir
from os.path import isfile, join
import random
from paths import *
total = np.zeros(8)

"""
Calculate sum of emotions in CK+ or Google dataset
Used for calculating class weights
"""

counter = 0
for numpy in (Path(PATH_NUMPY_CK).glob('*.npy')):
    e = np.load(numpy, allow_pickle=True)
    total = np.add(e[1], total)
    counter += 1

print(total)
