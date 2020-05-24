from __future__ import absolute_import, division, print_function, unicode_literals
import IPython.display as display
from PIL import Image  # Pillow Pil
import numpy as np
import os
import sys
import argparse
from pathlib import Path, PurePath
from paths import *
from facenet_pytorch import MTCNN, InceptionResnetV1

from config import *
import torchvision


def bool_action(action_name):
    result = ''
    while (result != 'y') and (result != 'n'):
        result = input(action_name + "? - y/n\n")

    if result == 'y':
        result = True
    elif result == 'n':
        result = False
    return result


# Config
SAVE_NPY = bool_action("Save npy file")

total_emo = np.zeros(len(EMOTION_DECLARATION))
USE_FACE_DETECT = bool_action("face detect?")
if USE_FACE_DETECT:
    face_detect = MTCNN(image_size=IMG_SIZE, select_largest=False, post_process=False)


def calc_emo_vector(emotion_name):
    result = np.array([1 if emo == emotion_name else 0 for emo in EMOTION_DECLARATION])
    return result


def create_vectors(emotion_name, img_fullpath):
    global total_emo
    print(img_fullpath)
    img = Image.open(img_fullpath).convert("RGB")  # RGB needed for face_detect
    x, y = img.size

    if x > MIN_PIC_SIZE and y > MIN_PIC_SIZE:  # face detection error if too small !!!
        img = face_detect(img)
        if type(img) != type(None):
            img = np.transpose(img.numpy().astype('uint8'), (1, 2, 0))  # unit8 + transform
            # img = Image.fromarray(img, 'RGB')
            # img.show()

            emo_vector = calc_emo_vector(emotion_name)
            total_emo = np.add(emo_vector, total_emo)

            img_fullpath = img_fullpath.stem[0:20]  # shorten name to 20 chars
            # Save img
            if (SAVE_NPY):
                npy_filename = str(Path(PATH_NUMPY_GOOGLE, str(emotion_name) + "_" + str(img_fullpath)))
                np.save(npy_filename, np.array((img, emo_vector)))
        else:
            print("No face found on", img_fullpath)
            with open("faceless_google.txt", "a+") as f:
                f.write(str(img_fullpath)+"\n")
    else:
        print("Too small picture", img_fullpath)
        with open("too_small.txt", "a+") as f:
            f.write(str(img_fullpath)+"\n")


# you can iterate glob only once
for emotion_name in EMOTION_DECLARATION:
    for img_fullpath in Path(PATH_DATASET_GOOGLE, emotion_name).glob('*.png'):
        create_vectors(emotion_name, img_fullpath)

np.savetxt('total_google_emo.txt', total_emo)
