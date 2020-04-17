
from __future__ import absolute_import, division, print_function, unicode_literals
import torch.optim as optim
import torch.nn.functional as F
import IPython.display as display
from PIL import Image  # Pillow Pil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path, PurePath
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import skimage.io as io
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn as nn
import torch.nn.functional as F


def imageShowImage():
    plt.figure()

    def imshow(img):
        # unnormalize
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


def dataset_itterating():
    for i_batch, sample_batched in enumerate(train_dataset):
        image, emotion = sample_batched
        print(image)
        if (i_batch == 0):
            imshow(image)
            plt.pause(0.001)
            break
