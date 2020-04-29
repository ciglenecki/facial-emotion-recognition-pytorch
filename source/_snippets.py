
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
# https://discuss.pytorch.org/t/multi-label-classification-in-pytorch/905/44


model = nn.Linear(20, 5)  # predict logits for 5 classes
x = torch.randn(1, 20)
y = torch.tensor([[1., 0., 1., 0., 0.]])  # get classA and classC as active

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-1)

for epoch in range(20):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print('Loss: {:.3f}'.format(loss.item()))

if i % BATCH_PRINT == BATCH_PRINT-1:
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            face, emotions = batch

            print(tabulate([["Outputs", outputs], ["Emotions", emotions]]))

            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / BATCH_PRINT))
            train_losses.append(running_loss/len(train_loader))
            test_losses.append(running_loss/len(train_loader))
            running_loss = 0
            model.train()
            break
