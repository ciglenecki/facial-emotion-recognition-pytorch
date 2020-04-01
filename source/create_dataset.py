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

emotion_total = np.array([0, 0, 0, 0, 0, 0, 0, 0])
IMG_WIDTH = 32
IMG_HEIGHT = 490

path_project = "/home/matej/projects/fer-projekt/"
path_dataset_string = "/home/matej/projects/fer-projekt/ck+/"

path_dataset = Path(path_dataset_string)
path_emotions = Path(path_dataset, "emotions")
path_facs = Path(path_dataset, "facs")
path_images = Path(path_dataset, "images")
path_landmarks = Path(path_dataset, "landmarks")
path_numpy = Path(path_project, "numpy")

filepaths_emotions = path_emotions.glob("*/*/*")
filepaths_facs = path_facs.glob("*/*/*")
filepaths_images = path_images.glob("*/*/*.png")
filepaths_landmarks = path_landmarks.glob("*/*/*")
filepaths_numpy = sorted(path_numpy.glob("*.npy"))

mtcnn = MTCNN(image_size=IMG_WIDTH, select_largest=False, post_process=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
tensor_to_PIL = transforms.ToPILImage()


def npy_to_sample(npy_filepath):
    numpy_sample = np.load(npy_filepath, allow_pickle=True)
    numpy_image = numpy_sample[0]
    numpy_emotion = numpy_sample[1]
    image = Image.fromarray(numpy_image).convert('LA').convert('RGB')
    emotion = numpy_emotion
    return image, emotion


def facedetect_to_pil(image):
    image = np.array(image)
    # Stacks r,g,b into rgb
    return Image.fromarray(np.transpose(image, (1, 2, 0)).astype(np.uint8))


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
    """Apply facedetect to image.

    Args:
        None
    """

    def __call__(self, image):
        image = mtcnn(image)
        image = facedetect_to_pil(image)
        return image


class MyDataset(Dataset):
    def __init__(self, filepaths_numpy, transform_image=None, transform_sample=None):
        self.filepaths_numpy = filepaths_numpy
        self.transform_image = transform_image
        self.transform_sample = transform_sample

    def __len__(self):
        return len(filepaths_numpy)

    def __getitem__(self, idx):
        global emotion_total
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_name = str(Path(self.filepaths_numpy[idx]))
        image, emotion = npy_to_sample(sample_name)
        emotion_total = emotion_total + np.array(emotion)

        # Image transformation
        if self.transform_image:
            image = self.transform_image(image)

        # Sample transformation
        sample = image, emotion
        if self.transform_sample:
            sample = self.transform_sample(sample)

        return image, emotion


dataset = MyDataset(filepaths_numpy,
                    transform_sample=transforms.Compose([
                    ]),
                    transform_image=transforms.Compose([
                        FaceDetect(),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]))

dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)


plt.figure()


def imshow(img):
    # unnormalize
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# # Dataset itterating
# for i_batch, sample_batched in enumerate(train_dataset):
#     image, emotion = sample_batched
#     print(image)
#     if (i_batch == 0):
#         imshow(image)
#         plt.pause(0.001)
#         break


train_split = .2
shuffle_dataset = True

train_size = int(train_split * len(dataloader))
test_size = len(dataloader) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                          shuffle=True, num_workers=2)


resnet18 = models.resnet18(pretrained=True,  progress=True)
resnet18.eval()
lr = 0.1


def get_model():
    model = resnet18
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    return model, optimizer


model, optimizer = get_model()
criterion = nn.NLLLoss()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, 32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


PATH = './cifar_net.pth'
net = Net()

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # inputs = inputs.squeeze(0)
        # $labels = labels.squeeze(0)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
