from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

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
from config import *

i = 0
PATH_NUMPY = PATH_NUMPY_CK  # numpy_google | numpy
PATH_TEST = Path(PATH_NUMPY, "test")


files_numpy = [f for f in listdir(PATH_TEST) if isfile(join(PATH_TEST, f))]

for numpy in files_numpy:
    shutil.move(join(PATH_TEST, numpy), PATH_NUMPY)

files_numpy = [f for f in listdir(
    PATH_NUMPY) if isfile(join(PATH_NUMPY, f))]

random_files = random.sample(
    files_numpy, int(len(files_numpy)*TEST_SPLIT))

for numpy in random_files:
    full_path = os.path.join(PATH_NUMPY, numpy)
    os.rename(full_path,
              os.path.join(PATH_TEST, numpy))
    i += 1

print("Created ", i, " files in folder ", PATH_TEST)
