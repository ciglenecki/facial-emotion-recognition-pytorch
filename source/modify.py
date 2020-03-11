from __future__ import absolute_import, division, print_function, unicode_literals
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
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import skimage.io as io


emotion_declaration = [
    "neutral",
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "sadness",
    "surprise",
]

IMG_WIDTH = 640
IMG_HEIGHT = 490

path_project = "/home/matej/projects/fer-projekt/"
path_dataset_string = "/home/matej/projects/fer-projekt/ck+/"

path_dataset = Path(path_dataset_string)
path_emotions = Path(path_dataset, "emotions")
path_facs = Path(path_dataset, "facs")
path_images = Path(path_dataset, "images")
path_landmarks = Path(path_dataset, "landmarks")
path_numpy = Path(path_project, "numpy")


# filepaths_emotions = tf.data.Dataset.list_files(str(str(path_emotions) + '*/*/*'))
filepaths_emotions = path_emotions.glob("*/*/*")
filepaths_facs = path_facs.glob("*/*/*")
filepaths_images = path_images.glob("*/*/*.png")
filepaths_landmarks = path_landmarks.glob("*/*/*")
filepaths_numpy = sorted(path_numpy.glob("*.npy"))


# plt.figure()


class MyDataset(Dataset):
    def __init__(self, filepaths_numpy, transform=None):
        self.filepaths_numpy = filepaths_numpy
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            print("test")
            idx = idx.tolist()

        img_name = str(Path(self.filepaths_numpy[idx]))
        numpy_object = np.load(img_name, allow_pickle=True)
        numpy_image = numpy_object[0]
        numpy_emotion = numpy_object[1]
        image = torch.from_numpy(numpy_image)
        emotion = torch.from_numpy(numpy_emotion)
        # x = torch.from_numpy(numpy_image).float()
        sample = {'image': image, 'emotion': emotion}
        if self.transform:
            sample['image'] = self.transform(
                np.array(Image.fromarray(numpy_image).convert('LA'))
        return sample

    def __len__(self):
        return len(filepaths_numpy)


transform=transforms.Compose([
    transforms.RandomCrop(60),
    transforms.RandomHorizontalFlip()
])


dataset=MyDataset(filepaths_numpy, transform)
dataloader=DataLoader(dataset, batch_size=4, num_workers=4)


plt.figure()

for i_batch, sample_batched in enumerate(dataloader):
    image=Image.fromarray(sample_batched['image'].numpy()[0]).convert('LA')
    print(i_batch, image)
    plt.imshow(image)
    plt.pause(0.001)

    if (i_batch == 4):
        break
    # observe 4th batch and stop

plt.show()

# Python Tensor to PIL
# Image.fromarray(sample_batched['image'].numpy()[0])
