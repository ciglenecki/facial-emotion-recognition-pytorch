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
from torch.optim import lr_scheduler
import sys
from tabulate import tabulate
import copy

"""
module_name
package_name
ClassName
method_name
ExceptionName
function_name
GLOBAL_CONSTANT_NAME
global_var_name
instance_var_name
function_parameter_name
local_var_name
"""


def bool_action(action_name):
    result = ''
    while (result != 'y') and (result != 'n'):
        result = input(action_name + "? - y/n\n")

    if result == 'y':
        result = True
    elif result == 'n':
        result = False
    return result


def number_action(number_name):
    number = ''
    while (not isinstance(number, int) and not isinstance(number, float)):
        number = input(number_name + "?\n")
        number = float(number)
    return number


# Paths
PATH_PROJECT = Path.cwd()

# DATASET
PATH_DATASET = Path(PATH_PROJECT, "ck+")

PATH_EMOTIONS = Path(PATH_DATASET, "emotions")
PATH_FACS = Path(PATH_DATASET, "facs")
PATH_IMAGES = Path(PATH_DATASET, "images")
PATH_LANDMARKS = Path(PATH_DATASET, "landmarks")

FILEPATHS_EMOTIONS = PATH_EMOTIONS.glob("*/*/*")
FILEPATHS_FACS = PATH_FACS.glob("*/*/*")
FILEPATHS_IMAGES = PATH_IMAGES.glob("*/*/*.png")
FILEPATHS_LANDMARKS = PATH_LANDMARKS.glob("*/*/*")
# SOURCE
PATH_SOURCE = Path(PATH_PROJECT, "source")

# NUMPY
PATH_NUMPY = Path(PATH_PROJECT, "numpy")
PATH_NUMPY_VALIDATION = Path(PATH_NUMPY, "validation")
FILEPATHS_NUMPY = sorted(PATH_NUMPY.glob("*.npy"))
PATH_NUMPY_VALIDATION = sorted(PATH_NUMPY_VALIDATION.glob("*.npy"))
# MODELS
PATH_MODELS = Path(PATH_PROJECT, "models")
PATH_MODEL_SAVE = Path(PATH_MODELS, 'resnet34.pth')
PATH_MODEL_STATE_CK_SAVE = Path(PATH_MODELS, 'resnet34_ck.pth')

# Global variables
EMOTION_DECLARATION = [
    "neutral",
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "sadness",
    "surprise",


]


# Configuration
IMG_SIZE = 224

DATASET_DROP_RATE = number_action("Dataset drop rate")
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2


DO_TRAIN_MODEL = bool_action("Train model")
LEARNING_RATE = 0.001
EPOCHS = int(number_action("Epochs"))
BATCH_SIZE = 16
BATCH_PRINT = 15
WEIGHTS = [2843.395, 494.659, 111.962, 423.327, 270.119, 648.553, 264.989, 645.996]
WEIGHTS = torch.from_numpy(np.amin(WEIGHTS)/np.array(WEIGHTS))


print(WEIGHTS)
torch.set_printoptions(sci_mode=False)


def npy_to_sample(npy_filepath):
    numpy_sample = np.load(npy_filepath, allow_pickle=True)
    numpy_image = numpy_sample[0]
    numpy_emotion = numpy_sample[1]
    image = Image.fromarray(numpy_image).convert('LA').convert('RGB')
    emotion = numpy_emotion
    return image, emotion


def tensor_to_image(tensor):
    numpy_sample = np.load(npy_filepath, allow_pickle=True)
    numpy_image = numpy_sample[0]
    numpy_emotion = numpy_sample[1]
    image = Image.fromarray(numpy_image).convert('LA').convert('RGB')
    emotion = numpy_emotion
    return image, emotion


def set_train_val_size(train_split, VAL_SPLIT):
    """ Set size for training and validation set
    Args:
        train_split [0,1] - percentage of train images
        VAL_SPLIT [0,1] - percentage of validation images
    """

    if (train_split + VAL_SPLIT > 1.0):
        sys.exit("Train size + validation size is bigger dataset")

    dataset_size = len(dataset)
    train_size = int(np.floor(train_split * dataset_size))
    val_size = dataset_size - train_size

    return train_size, val_size


def CrossEntropyLossSoftTarget(pred, soft_targets, verbose=False):

    def batch_tensor_value(tensor):
        return (torch.sum(tensor, 0)/len(tensor)).data

    logsoftmax = nn.LogSoftmax(dim=1)  # log(softmax(x))
    softmax = nn.Softmax(dim=1)  # log(softmax(x))

    plain_loss = - soft_targets * logsoftmax(pred)
    weighted_loss = WEIGHTS * (plain_loss)
    result = torch.mean(
        torch.sum(
            weighted_loss, 1
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
            ["Weights", WEIGHTS.data],
            ["Soft targets", batch_soft_targets],
            ["Prediction probability", batch_pred_prob],
            ["Prediction LogSoftMax", batch_pred_logsoftmax],
            ["Loss", batch_plain_loss],
            ["Loss weighted (res)", batch_weighted_loss]
        ], tablefmt="github")

        print(table)

    return result


def load_model():
    if os.path.isfile(str(PATH_MODEL_SAVE)):
        resnet34 = torch.load(str(PATH_MODEL_SAVE))
    else:
        resnet34 = models.resnet34(pretrained=True,  progress=True)
        torch.save(resnet34, str(PATH_MODEL_SAVE))

    return resnet34


def get_model():
    model = load_model()
    for param in model.parameters():  # Param freezing
        param.requires_grad = False

    num_features = model.fc.in_features  # Model's last layer output number

    model.fc = nn.Sequential(
        nn.Linear(num_features, len(EMOTION_DECLARATION)),
    )

    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    return model, optimizer


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[: 2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, emotion = sample['image'], sample['emotion']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image), torch.from_numpy(emotion)


class FaceDetect(object):

    def __init__(self, image_size):
        self.face_detect = MTCNN(image_size=image_size,
                                 select_largest=False, post_process=False)

    def __call__(self, image):
        image = self.face_detect(image)
        image = self.facedetect_to_PIL(image)
        return image

    def facedetect_to_PIL(self, image):
        image = np.array(image)
        # Stacks r,g,b into rgb
        # (224, 224, 3)
        return Image.fromarray(np.transpose(image, (1, 2, 0)).astype(np.uint8))


class FERDataset(Dataset):
    def __init__(self, filepaths_numpy, transform_image=None, transform_emotion=None):
        self.filepaths_numpy = filepaths_numpy
        self.transform_image = transform_image
        self.transform_emotion = transform_emotion
        self.emotion_total_count = np.zeros(len(EMOTION_DECLARATION))

    def __len__(self):
        return len(self.filepaths_numpy)

    def __getitem__(self, idx):

        sample_name = str(Path(self.filepaths_numpy[idx]))
        image, emotion = npy_to_sample(sample_name)

        if self.transform_image:
            image = self.transform_image(image)

        if self.transform_emotion:
            emotion = emotion.astype(np.float)

        self.emotion_total_count = self.emotion_total_count + np.array(emotion)
        image, emotion
        return image, emotion

    def emotion_total_count_scaled(self):
        return (self.emotion_total_count/np.max(self.emotion_total_count))


dataset = FERDataset(filepaths_numpy=FILEPATHS_NUMPY,
                     transform_emotion=True,
                     transform_image=transforms.Compose([
                         FaceDetect(IMG_SIZE),
                         transforms.Resize(IMG_SIZE),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
                     ]))

dataset_length = int(len(dataset) * (1 - DATASET_DROP_RATE))
dataset_length_drop = len(dataset) - dataset_length
dataset, _ = torch.utils.data.random_split(
    dataset, [dataset_length, dataset_length_drop])


train_size, val_size = set_train_val_size(TRAIN_SPLIT, VAL_SPLIT)

print("Train, val:", train_size, val_size)
print("Dataset: ", len(dataset))


train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                           shuffle=True, drop_last=True)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                         shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                          shuffle=True, drop_last=True)


model, optimizer = get_model()
loss_func = CrossEntropyLossSoftTarget
train_losses, val_losses = [], []

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

softmax = nn.Softmax(dim=0)

max_acc = 0

if (DO_TRAIN_MODEL):
    print("\nStarting model training...\n")

    for epoch in range(EPOCHS):

        train_loss = 0.0
        batch_loss = 0

        val_loss = 0
        i = 0
        ############ TRAIN ######################

        model.train()
        for batch in train_loader:

            if(i == int(len(train_loader)/BATCH_SIZE) - 1):
                verbose = True
            else:
                verbose = False

            i += 1

            face, emotions = batch
            optimizer.zero_grad()
            outputs = model(face)  # face: batchsize x 3 x 244 x 244
            emotions = emotions.type_as(outputs)
            loss = loss_func(pred=outputs.float(), soft_targets=emotions.float(), verbose=verbose)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
            train_loss += loss.item() * emotions.size(0)  # has to collect all loses

            if verbose:
                print("Batch("+str(BATCH_SIZE)+"): "+str(i*BATCH_SIZE)+"/"+str(len(train_dataset)))

                print("batch_loss", batch_loss/BATCH_SIZE)

                print()

                batch_loss = 0
        exp_lr_scheduler.step()
        train_loss = train_loss / train_size
        train_losses.append(train_loss)
        print("\n[Epoch:", epoch, ", train_loss:", train_loss, "]\n")

        ############ VAL ACC ######################
        model.eval()
        epoch_accs = []
        with torch.no_grad():

            for batch in val_loader:
                face, emotions = batch
                optimizer.zero_grad()
                emotions = emotions.squeeze(0)

                outputs = model(face).squeeze(0)
                outputs = softmax(outputs)

                acc = (2 - torch.sum(torch.abs(emotions - outputs))) / 2
                epoch_accs.append(acc)

        epoch_acc = 100 * sum(epoch_accs) / len(epoch_accs)
        print("Epoch", epoch, "acc", epoch_acc)
        if(epoch_acc > max_acc):
            max_acc = epoch_acc
            best_model = copy.deepcopy(model.state_dict())

    print('Finished training...')
    torch.save(best_model, str(PATH_MODEL_STATE_CK_SAVE))

    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()

    print('Trained model saved to: ', str(PATH_MODEL_STATE_CK_SAVE))


model, _ = get_model()
model.load_state_dict(torch.load(str(PATH_MODEL_STATE_CK_SAVE)))


correct = 0
total = 0
j = 0
model.eval()
list_acc = []
with torch.no_grad():
    for batch in test_loader:
        face, emotions = batch
        optimizer.zero_grad()
        emotions = emotions.squeeze(0)

        outputs = model(face).squeeze(0)
        outputs = softmax(outputs)
        topk, indices = torch.topk(outputs, k=2)
        # outputs = torch.zeros(len(EMOTION_DECLARATION)).scatter(0, indices, topk)
        print(emotions)
        print(outputs)

        # 2 = maximum mistake
        acc = (2 - torch.sum(torch.abs(emotions - outputs))) / 2
        list_acc.append(acc)

print('Total acc', 100 * sum(list_acc) / len(list_acc))
