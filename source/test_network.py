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
from model import *
from paths import *
from config import *
model, _, _ = get_model()
model.load_state_dict(torch.load(str(Path(PATH_MODELS, 'resnet50_trained_acc_55.51374188840743_drop_0.0_epoch_44_lr_0.01_rLq7pt36.pth'))))

transform_image_val = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225])
])


class FERDataset(Dataset):
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


test_dataset = FERDataset(filepaths_numpy=FILEPATHS_NUMPY_TEST,
                          transform_emotion=True,
                          transform_image=transform_image_val)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)

correct = 0
total = 0
j = 0
model.eval()
list_acc = []
softmax = nn.Softmax(dim=0)
with torch.no_grad():
    for batch in test_loader:
        face, emotions = batch

        emotions = emotions.squeeze(0)

        outputs = model(face).squeeze(0)
        outputs = softmax(outputs)
        # outputs = torch.zeros(len(EMOTION_DECLARATION)).scatter(0, indices, topk)
        # print(emotions)
        # print(outputs)
        # 2 = maximum mistake
        acc = (2 - torch.sum(torch.abs(emotions - outputs))) / 2
        list_acc.append(acc)

print('Total acc', 100 * sum(list_acc) / len(list_acc))
