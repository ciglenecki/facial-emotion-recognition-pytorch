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
import sys


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

# Paths
PATH_PROJECT = Path("/home/matej/1-projects/fer-projekt/")
PATH_DATASET = Path(PATH_PROJECT, "ck+")
PATH_EMOTIONS = Path(PATH_DATASET, "emotions")
PATH_FACS = Path(PATH_DATASET, "facs")
PATH_IMAGES = Path(PATH_DATASET, "images")
PATH_LANDMARKS = Path(PATH_DATASET, "landmarks")

PATH_NUMPY = Path(PATH_PROJECT, "numpy")
PATH_NUMPY_VALIDATION = Path(PATH_NUMPY, "validation")

FILEPATHS_EMOTIONS = PATH_EMOTIONS.glob("*/*/*")
FILEPATHS_FACS = PATH_FACS.glob("*/*/*")
FILEPATHS_IMAGES = PATH_IMAGES.glob("*/*/*.png")
FILEPATHS_LANDMARKS = PATH_LANDMARKS.glob("*/*/*")
FILEPATHS_NUMPY = sorted(PATH_NUMPY.glob("*.npy"))
PATH_NUMPY_VALIDATION = sorted(PATH_NUMPY_VALIDATION.glob("*.npy"))


PATH_MODEL_SAVE = './resnet18.pth'
PATH_MODEL_STATE_CK_SAVE = './resnet18_ck.pth'


# Global variables
EMOTION_DECLARATION = [
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

TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.3

DO_TRAIN_MODEL = True
LEARNING_RATE = 0.001
EPOCHS = 5
BATCH_SIZE = 4


def npy_to_sample(npy_filepath):
    numpy_sample = np.load(npy_filepath, allow_pickle=True)
    numpy_image = numpy_sample[0]
    numpy_emotion = numpy_sample[1]
    image = Image.fromarray(numpy_image).convert('LA').convert('RGB')
    emotion = numpy_emotion
    print
    return image, emotion


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

        h, w = image.shape[:2]
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
        return Image.fromarray(np.transpose(image, (1, 2, 0)).astype(np.uint8))


class MyDataset(Dataset):
    def __init__(self, filepaths_numpy, transform_image=None, transform_emotion=None):
        self.filepaths_numpy = filepaths_numpy
        self.transform_image = transform_image
        self.transform_emotion = transform_emotion
        self.emotion_total_count = np.zeros(len(EMOTION_DECLARATION))

    def __len__(self):
        return len(self.filepaths_numpy)

    def __getitem__(self, idx):

        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        sample_name = str(Path(self.filepaths_numpy[idx]))
        image, emotion = npy_to_sample(sample_name)
        if self.transform_image:
            image = self.transform_image(image)

        if self.transform_emotion:
            emotion = emotion.astype(np.float)

        self.emotion_total_count = self.emotion_total_count + np.array(emotion)
        image, emotion
        return image, emotion


dataset = MyDataset(filepaths_numpy=FILEPATHS_NUMPY,
                    transform_emotion=True,
                    transform_image=transforms.Compose([
                        FaceDetect(IMG_SIZE),
                        transforms.Resize(IMG_SIZE),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                            0.229, 0.224, 0.225])
                    ]))


# def my_collate(batch):

#     images = [item[0] for item in batch]
#     emotions = [item[1] for item in batch]
#     emotions = torch.LongTensor(emotions)
#     for image in images:
#         print(image.size())

#     for emotion in emotions:
#         print(emotion.size())
#     return images, emotions


def set_train_val_size(train_split, validation_split):
    """ Set size for training and validation set
    Args:
        train_split [0,1] - percentage of train images
        validation_split [0,1] - percentage of validation images
    """
    if (train_split + validation_split > 1.0):
        sys.exit("Train size + validation size is bigger dataset")

    dataset_size = len(dataset)
    train_size = int(np.floor(train_split * dataset_size))
    validation_size = dataset_size - train_size

    return train_size, validation_size


train_size, validation_size = set_train_val_size(TRAIN_SPLIT, VALIDATION_SPLIT)

print("Train and validation size: ", train_size, validation_size)
print("Dataset size: ", len(dataset))


train_dataset, validation_dataset = torch.utils.data.random_split(
    dataset, [train_size, validation_size])


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                           shuffle=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                          shuffle=True, drop_last=True)


resnet18 = models.resnet18(pretrained=True,  progress=True)
torch.save(resnet18, PATH_MODEL_SAVE)


def get_model():
    model = resnet18

    # Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
    feature_extract = True

    # Freezing
    for param in model.parameters():
        param.requires_grad = False

    # In fetaures are previous in_features of resnet
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, len(EMOTION_DECLARATION)),
        nn.Softmax(dim=1)
    )

    print(model.fc)
    print(model.fc.parameters())

    # Recall that after loading the pretrained model, but before reshaping, if feature_extract=True we manually set all of the parameter’s .requires_grad attributes to False. Then the reinitialized layer’s parameters have .requires_grad=True by default. So now we know that all parameters that have .requires_grad=True should be optimized. Next, we make a list of such parameters and input this list to the SGD algorithm constructor.

    # mode.parameters()?
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return model, optimizer


model, optimizer = get_model()
model.train()
loss_func = nn.BCEWithLogitsLoss()
train_losses, test_losses = [], []
i = 0

if (DO_TRAIN_MODEL):
    print("Starting model training...")

    for epoch in range(EPOCHS):
        running_loss = 0.0

        for batch in train_loader:
            face, emotions = batch
            i += 1
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(face)
            emotions = emotions.type_as(outputs)
            loss = loss_func(outputs.float(), emotions.float())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # print(outputs)

            # print every 10 mini-batches
            if i % 10 == 10-1:
                model.eval()
                with torch.no_grad():
                    for batch in train_loader:
                        face, emotions = batch

                        print(emotions)
                        print(outputs)

                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 10))
                        running_loss = 0.0
                        train_losses.append(running_loss/len(train_loader))
                        test_losses.append(running_loss/len(train_loader))
                        running_loss = 0
                        model.train()

    print('Finished training...')
    torch.save(model.state_dict(), PATH_MODEL_STATE_CK_SAVE)

    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()

    print('Trained model saved to: ', PATH_MODEL_STATE_CK_SAVE)

model = resnet18
model.load_state_dict(torch.load(PATH_MODEL_STATE_CK_SAVE))


def correct_factor(truth, predicted):
    """From 0 to 1 how similar are truth and prediction
    """
    print(truth)
    predicted_normalized = []
    # print(predicted)
    print(predicted.numpy())
    for p in predicted.numpy()[0]:
        predicted_normalized.append((p+1)/2)
    # print(predicted_normalized)
    # for t in truth:


correct = 0
total = 0
# model.eval()
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        correct_factor(labels, outputs)
        # total += labels.size(0)
        # correct += (predicted == labels).sum().item()


print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
