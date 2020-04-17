from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import sys
from pathlib import Path, PurePath
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

i = 0

path_project = "/home/matej/1-projects/fer-projekt"
path_numpy = join(path_project, "numpy")
path_test = join(path_numpy, "test")

print(str(path_project), str(path_numpy), str(path_test))
test_split = 0.1

# test -> Original
files_numpy = [f for f in listdir(
    path_test) if isfile(join(path_test, f))]

for numpy in files_numpy:
    shutil.move(join(path_test, numpy), path_numpy)


files_numpy = [f for f in listdir(
    path_numpy) if isfile(join(path_numpy, f))]

random_files = random.sample(
    files_numpy, int(len(files_numpy)*test_split))

for numpy in random_files:
    full_path = os.path.join(path_numpy, numpy)
    os.rename(full_path,
              os.path.join(path_test, numpy))
    i += 1
print("Created ", i, " files in folder ", path_test)
