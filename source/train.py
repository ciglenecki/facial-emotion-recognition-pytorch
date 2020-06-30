from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import cv2
import sys
from pathlib import Path, PurePath
import matplotlib.pyplot as plt
import numpy as np
import copy
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models, transforms
from torch.optim import lr_scheduler
from tabulate import tabulate
from paths import *
from config import *
from model import *
from config_train import *
import faulthandler
import gc
ImageFile.LOAD_TRUNCATED_IMAGES = True
faulthandler.enable(all_threads=True)


def write_model_log(filename, train_losses, val_losses):

    filename = str(PATH_MODEL_TMP) + filename+".log"
    with open(filename, "w") as model_log:
        for attribute in ["DATASET_DROP_RATE", "DO_USE_SCHEDULER", "LEARNING_RATE", "EPOCHS", "BATCH_SIZE", "NUM_WORKERS", "IMG_SIZE", "TEST_SPLIT", "TRAIN_SPLIT", "VAL_SPLIT", "WEIGHTS_CK", "WEIGHTS_GOOGLE", "WEIGHTS", "train_losses", "val_losses", "OPTIMIZER", "GOOGLE_TRAIN_SPLIT", "CK_TRAIN_SPLIT"]:
            line = [str(attribute), str(globals()[attribute])]
            model_log.write(" ".join(line) + "\n")
    model_log.close()
    return filename


def get_model_name(acc=0, epoch=0):
    suffix = "_"+MODEL_SUFFIX
    dr = "_drop_"+str(DATASET_DROP_RATE)
    epoch = "_epoch_"+str(epoch)
    lr = "_lr_" + str(LEARNING_RATE)
    acc = "_acc_" + str(acc)
    goo_split = "_goosplit_"+str(GOOGLE_TRAIN_SPLIT)
    ck_split = "_cksplit_"+str(CK_TRAIN_SPLIT)

    return acc+dr+epoch+lr+suffix+goo_split+ck_split


def save_tmp_model(model, filename):
    filename = str(PATH_MODEL_TMP)+filename+".pth"
    torch.save(model, filename)


def save_model(model, filename):
    filename = str(PATH_MODEL_STATE)+filename+".pth"
    torch.save(model, filename)
    print('Trained model saved to: ', filename)


def tensor_to_image(tensor):
    numpy_sample = np.load(npy_filepath, allow_pickle=True)
    numpy_image = numpy_sample[0]
    numpy_emotion = numpy_sample[1]
    image = Image.fromarray(numpy_image).convert('L').convert('RGB')
    emotion = numpy_emotion
    return image, emotion


def set_train_val_size(TRAIN_SPLIT, VAL_SPLIT):
    """ Set size for training and validation set
    Args:
        TRAIN_SPLIT [0,1] - percentage of train images
        VAL_SPLIT [0,1] - percentage of validation images
    """

    if (TRAIN_SPLIT + VAL_SPLIT > 1.0):
        sys.exit("Train size + validation size is bigger dataset")

    dataset_size = len(dataset)
    train_size = int(np.floor(TRAIN_SPLIT * dataset_size))
    val_size = dataset_size - train_size

    return train_size, val_size


def calc_batch_acc(outputs, emotions):
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
    sums = torch.sum(torch.abs(emotions - outputs), dim=1)
    return float(torch.mean((2 - sums)/2))


class NPYDataset(Dataset):
    def __init__(self, filepaths_numpy, transform_image=None, transform_emotion=None):
        self.filepaths_numpy = filepaths_numpy
        self.transform_image = transform_image
        self.transform_emotion = transform_emotion

    def __len__(self):
        return len(self.filepaths_numpy)

    def __getitem__(self, idx):

        sample_name = str(Path(self.filepaths_numpy[idx]))
        image, emotion = self.npy_to_sample(sample_name)

        if self.transform_image:
            image = self.transform_image(image)

        if self.transform_emotion:
            emotion = emotion.astype(np.float)

        return image, emotion

    def npy_to_sample(self, npy_filepath):
        numpy_sample = np.load(npy_filepath, allow_pickle=True)
        numpy_image = numpy_sample[0]
        numpy_emotion = numpy_sample[1]
        image = Image.fromarray(numpy_image).convert('L').convert('RGB')
        emotion = numpy_emotion
        return image, emotion


transform_image_train = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomAffine(degrees=(-AUG_DEGREE, AUG_DEGREE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225])
])
transform_image_val = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225])
])
dataset = NPYDataset(filepaths_numpy=FILEPATHS_NUMPY,
                     transform_emotion=True,
                     transform_image=transform_image_train)


dataset_length = int(len(dataset) * (1 - DATASET_DROP_RATE))
dataset_length_drop = len(dataset) - dataset_length
dataset, _ = torch.utils.data.random_split(
    dataset, [dataset_length, dataset_length_drop])


train_size, val_size = set_train_val_size(TRAIN_SPLIT, VAL_SPLIT)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

val_dataset.transform_image = transform_image_val


print("Dataset:", len(dataset), "\n")
print("\nTrain, val:", train_size, val_size)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)


model = get_model()
optimizer, exp_lr_scheduler = get_optimizer(model)
loss_func = CrossEntropyLossSoftTarget

train_losses, val_losses = [], []
best_acc = 0
best_epoch = 0


if (DO_TRAIN_MODEL):
    print("\nStarting model training...\n")
    for epoch in range(EPOCHS):

        train_loss = 0.0
        batch_loss = 0
        val_loss = 0
        epoch_accs = []
        i = 0
        verbose = False
        # TRAIN ##################################

        model.train()
        for batch in train_loader:
            i += 1
            if(i == int(len(train_dataset)/BATCH_SIZE)):
                verbose = True

            face, emotions = batch
            optimizer.zero_grad()
            outputs = model(face)  # face: BATCH_SIZE x 3 x 244 x 244
            emotions = emotions
            loss = loss_func(pred=outputs.float(), soft_targets=emotions.float(), weights=WEIGHTS, verbose=verbose)
            loss.backward()
            optimizer.step()

            train_loss += float(loss) * int(emotions.size(0))  # emotion.   size(0) = batch size, dont use batch size because it might not be 16

            if verbose:
                print()
                print("Batch("+str(BATCH_SIZE)+"): "+str(i*BATCH_SIZE)+"/" + str(len(train_dataset)))W
                print("batch_loss", float(loss)/BATCH_SIZE)
                print()
                verbose = False

        train_loss = train_loss / train_size
        train_losses.append(train_loss)

        # VALIDATION-ACCURACY ##################################
        model.eval()
        i = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is not None:
                    i += 1
                    if(i == int(len(val_dataset)/BATCH_SIZE)):
                        verbose = True
                    else:
                        verbose = False

                    face, emotions = batch
                    optimizer.zero_grad()

                    outputs = model(face)
                    emotions = emotions

                    loss = loss_func(pred=outputs.float(),  soft_targets=emotions.float(),   weights=WEIGHTS, verbose=verbose)
                    val_loss += float(loss) * int(emotions.size(0))  # has to collect all loses

                    softmax = nn.Softmax(dim=1)
                    batch_acc = float(calc_batch_acc(outputs=softmax(outputs),  emotions=emotions))
                    epoch_accs.append(batch_acc)

        if(DO_USE_SCHEDULER):
            exp_lr_scheduler.step()

        val_loss = val_loss / val_size
        val_losses.append(val_loss)

        epoch_acc = (100 * sum(epoch_accs) / len(epoch_accs))

        if(epoch_acc > best_acc):
            best_acc = epoch_acc
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
            filename = get_model_name(acc=epoch_acc, epoch=epoch)
            save_tmp_model(best_model, filename)
            write_model_log(filename, train_losses, val_losses)

        print("\n[Epoch:", epoch, ",", "acc", epoch_acc, ", train_loss: ", train_loss, ",", "val_loss: ", val_loss, "lr: ", optimizer.param_groups[0]['lr'], "]\n\n")

    print('\nFinished with best_acc: ', best_acc)
    # Model saving
    filename = get_model_name(acc=best_acc, epoch=best_epoch)
    save_model(best_model, filename)
    write_model_log(filename, train_losses, val_losses)

    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.savefig(str(Path(PATH_VISUALS, filename+'.png')))
    print('Plot saved to: ', str(Path(PATH_VISUALS, filename+'.png')))
