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
from PIL import ImageFile
from paths import *
import random
import string
from config import *
from config_train import *


def CrossEntropyLossSoftTarget(pred, soft_targets, weights, verbose=False):

    def batch_tensor_value(tensor):
        return (torch.sum(tensor, dim=0)/len(tensor)).data

    logsoftmax = nn.LogSoftmax(dim=1)
    softmax = nn.Softmax(dim=1)

    plain_loss = - soft_targets * logsoftmax(pred)
    weighted_loss = weights * (plain_loss)
    result = torch.mean(
        torch.sum(
            weighted_loss, dim=1
        )
    )
    if (verbose):
        # Softmax and LogSoftmax are applied to 2D vectors, then batch_tensor_value takes average to 1D vector
        batch_soft_targets = batch_tensor_value(soft_targets)
        batch_pred_prob = batch_tensor_value(softmax(pred))
        batch_pred_logsoftmax = batch_tensor_value(logsoftmax(pred))

        batch_plain_loss = batch_tensor_value(plain_loss)
        batch_weighted_loss = batch_tensor_value(weighted_loss)

        table = tabulate([
            ["Images", soft_targets],
            ["Weights", weights.data],
            ["Soft targets", batch_soft_targets],
            ["Prediction probability", batch_pred_prob],
            ["Prediction LogSoftMax", batch_pred_logsoftmax],
            ["Loss", batch_plain_loss],
            ["Loss weighted (res)", batch_weighted_loss]
        ], tablefmt="github")

        print(table)

    return result


def load_model():
    if os.path.isfile(str(PATH_MODEL)):
        resnet50 = torch.load(str(PATH_MODEL))
    else:
        resnet50 = models.resnet50(pretrained=True,  progress=True)
        torch.save(resnet50, str(PATH_MODEL))

    return resnet50


def get_model():
    model = load_model()

    # Freeze all but last 3 layers
    for model_block in list(model.children())[:-3]:
        for param in model_block.parameters():
            param.requires_grad = False

    # Freeze whole resnet
    # for param in model.parameters():
    #     param.requires_grad = False

    num_features = model.fc.in_features  # Model's last layer output number

    model.fc = nn.Sequential(
        nn.Linear(num_features, len(EMOTION_DECLARATION)),
        # nn.Linear(num_features, len(EMOTION_DECLARATION)**2),
        # nn.BatchNorm1d(len(EMOTION_DECLARATION)**2),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(len(EMOTION_DECLARATION)**2, len(EMOTION_DECLARATION)),
    )

    return model


def get_optimizer(model):
    if OPTIMIZER == 0:
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    elif OPTIMIZER == 1:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=int(EPOCHS/4), gamma=0.1)
    return optimizer, exp_lr_scheduler
