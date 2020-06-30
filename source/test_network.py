from __future__ import absolute_import, division, print_function, unicode_literals
import torch.optim as optim
import torch.nn.functional as F
import IPython.display as display
from PIL import Image  # Pillow Pil
import numpy as np
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
import seaborn as sn
import pandas as pd

model = get_model()
model.load_state_dict(torch.load(str(Path(PATH_MODELS, "tmp", 'resnet50_trained_acc_72.98471995799356_drop_0.0_epoch_57_lr_0.0001_jRJ668Mq_goosplit_1_cksplit_1.pth'))))

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

        return image, emotion, sample_name

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
list_acc_binary = []
list_acc_two = []


confusion_matrix = torch.zeros(len(EMOTION_DECLARATION), len(EMOTION_DECLARATION))


cmt = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [
    0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]

softmax = nn.Softmax(dim=0)
softmax_dim1 = nn.Softmax(dim=1)
i = 0
with torch.no_grad():
    for batch in test_loader:

        i = i+1
        face, emotions, sample_name = batch

        outputs = model(face)

        emotions_nonsqueze = emotions
        outputs_nonsqueze = softmax_dim1(outputs)

        emotions = emotions.squeeze(0)
        outputs = outputs.squeeze(0)

        outputs = softmax(outputs)

        # outputs = torch.zeros(len(EMOTION_DECLARATION)).scatter(0, indices, topk)
        # print(emotions)
        # print(outputs)
        # 2 = maximum mistake
        # print(emotions)
        # print(outputs)

        acc = (2 - torch.sum(torch.abs(emotions - outputs))) / 2

        _, idx1 = torch.max(outputs, dim=0, keepdim=True)
        _, idx2 = torch.max(emotions, dim=0, keepdim=True)

        confusion_matrix[idx1, idx2] += 1

        acc_binary = int(idx1 == idx2)
        cmt[idx2][idx1] = cmt[idx2][idx1] + 1

        topk, indices = torch.topk(outputs, k=2)

        res = torch.zeros(len(outputs))
        topk = res.scatter(dim=0, index=indices, src=topk)

        acc_two = (2 - torch.sum(torch.abs(emotions - topk))) / 2

        list_acc_binary.append(acc_binary)
        list_acc.append(acc)
        list_acc_two.append(acc_two)
        if ((i % int((len(test_dataset)/20))) == 0):
            i = 0
            print(str(sample_name).split('/')[-1])
            print('truth', emotions)
            print('pred', outputs)

            print('acc_binary', acc_binary)
            print('acc', acc)
            print('acc_two', acc_two)
        # elif acc > 0.7:
        #     print(str(sample_name).split('/')[-1])
        #     print('truth', emotions)
        #     print('pred', outputs)

        #     print('acc_binary', acc_binary)
        #     print('acc', acc)
        #     print('acc_two', acc_two)


print('Total acc', 100 * sum(list_acc) / len(list_acc))
print('Binary acc', 100 * sum(list_acc_binary) / len(list_acc_binary))
print('Two acc', 100 * sum(list_acc_two) / len(list_acc_two))

# Per class accuracy
class_acc = (confusion_matrix.diag()/confusion_matrix.sum(1)).numpy()
print(class_acc)
print(confusion_matrix)

confusion_matrix = (confusion_matrix/torch.sum(confusion_matrix, dim=0)).numpy()
df_matrix = pd.DataFrame(confusion_matrix, EMOTION_DECLARATION, EMOTION_DECLARATION)
df_classes = pd.DataFrame(class_acc, EMOTION_DECLARATION)

sn.set(font_scale=1.2)  # for label size
ax_matrix = sn.heatmap(df_matrix, annot=True, cmap="Blues", annot_kws={"size": 16})  # font size


plt.show()
plt.savefig('matrix.png')

# x = input("Go to next plot")
# plt.close()
# ax_classes = sn.heatmap(df_classes.T, annot=True, cmap="Blues", annot_kws={"size": 16})  # font size
# plt.show()
