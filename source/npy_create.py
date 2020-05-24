from __future__ import absolute_import, division, print_function, unicode_literals
import IPython.display as display
from PIL import Image  # Pillow Pil
import numpy as np
import os
import sys
import argparse
from pathlib import Path, PurePath
from paths import *
from config import *
from facenet_pytorch import MTCNN, InceptionResnetV1
from misc_input import *
import torchvision
import matplotlib.pyplot as plt
import PIL.ImageOps
# Config

SAVE_NPY = bool_action("Save npy file")
NEUTRAL_EXISTS = True

USE_FACE_DETECT = bool_action("Apply facedetect")
if USE_FACE_DETECT:
    face_detect = MTCNN(image_size=IMG_SIZE, select_largest=False, post_process=False)


total_emo = np.zeros(len(EMOTION_DECLARATION))


def validate_emo_vector(emo_vector):
    if (np.min(emo_vector) < 0):
        print("Emo_vector has negative emotion", emo_vector)
        return - 1
    if sum(emo_vector) > 1:
        print("Emo_vector's sum is larger than 1: ", emo_vector)
        return - 1
    return emo_vector


def attr_to_prefolder(person, seq_num, file_type_name=None):

    if(file_type_name == "emotion"):
        pre_path = PATH_CK_EMOTIONS
    elif(file_type_name == "image"):
        pre_path = PATH_CK_IMAGES

    return Path(pre_path, person, seq_num)


def attr_to_filename(person, seq_num, seq_count_text, file_type_name=None):
    if(file_type_name == "emotion"):
        filename = "_".join([person, seq_num, seq_count_text, "emotion.txt"])

    elif(file_type_name == "image"):
        filename = "_".join([person, seq_num, seq_count_text + ".png"])

    prefolder = attr_to_prefolder(person, seq_num, file_type_name)
    return Path(prefolder, filename)


def filename_to_fullpath(filename):
    person, seq_num, seq_count_text, file_type = split_filename(filename)
    pre_path = attr_to_prefolder(person, seq_num, file_type)
    return Path(pre_path, filename)


def get_img_filenames(person, seq_num):
    """Returns img filenames for given person and his ordinal number sequence
    Args:
        person (string)
        seq_num (string)
        seq_count_text (string)
    """

    img_folder = attr_to_prefolder(person, seq_num, "image")
    img_batch_filenames = list(sorted(img_folder.glob("*.png"), reverse=True))
    return img_batch_filenames


def get_emotion(emo_filename):
    """Reads emotion value from emotion filename
    Filename is constructed from

    Args:
        person (string)
        seq_num (string)
        seq_count_text (string)
    """
    if os.path.isfile(emo_filename):
        emotion_file = open(str(emo_filename), "r",)
    else:
        return -1

    return int(float(emotion_file.readline()))


def split_filename(filename):
    """Return parts of a filename
    Args:
        filename (string) - S125_001_00000014_emotion.txt
    """
    filename = str(filename)
    if (filename.endswith(".png")):
        file_type = "image"
    if (filename.endswith("emotion.txt")):
        file_type = "emotion"

    filename_parts = filename.split("_")
    person = filename_parts[0]  # S154
    seq_num = filename_parts[1]  # 002
    seq_count_text = os.path.splitext(filename_parts[2])[0]  # 00000014
    return person, seq_num, seq_count_text, file_type


def fullpath_to_filename(filepath):
    filename_parts = str(filepath).split(os.path.sep)
    return filename_parts[len(filename_parts) - 1]


def calc_emo_vector(i, max_seq, emotion):

    p = (i-1) / (max_seq-1)
    emo_vector = np.zeros(len(EMOTION_DECLARATION))

    if (NEUTRAL_EXISTS):
        emo_vector[0] = round(1 - p, 3)
    emo_vector[emotion] = round(p, 3)

    return validate_emo_vector(emo_vector)


def create_vectors(person, seq_num, seq_count_text):

    emo_filename = attr_to_filename(person, seq_num, seq_count_text, "emotion")
    emotion = get_emotion(emo_filename)

    if not emotion == -1:
        img_batch_filenames = get_img_filenames(person, seq_num)
        """
        Find out arbitrary number of imgs in sequence
        Number of pictrues doesn't represent maximal needed
        1, 2, 4 => should be 4, not 3
        """
        _, _, max_seq, _ = split_filename(str(img_batch_filenames[0]))
        max_seq = int(max_seq)

        """Find img and cooresponding emotion
        Save it as .npy file
        """

        for i, img_filename in enumerate(img_batch_filenames, start=0):

            _, _, i, _ = split_filename(str(img_filename))
            i = int(i)
            img_fullpath = filename_to_fullpath(img_filename)
            img = Image.open(img_fullpath).convert("RGB")  # RGB needed for face_detect
            img = face_detect(img)
            # img = Image.fromarray(img, 'RGB')
            # img.show()

            if type(img) != type(None):
                img = np.transpose(img.numpy().astype('uint8'), (1, 2, 0))  # unit8 + transform

                emo_vector = calc_emo_vector(i, max_seq, emotion)

                if emo_vector is not -1:

                    global total_emo

                    total_emo = np.add(emo_vector, total_emo)
                    # Save img
                    if (SAVE_NPY):
                        npy_filename = str(Path(PATH_CK_NUMPY, str(person
                                                                   + "_"
                                                                   + seq_num
                                                                   + "_"
                                                                   + str(i).zfill(8))))
                        np.save(npy_filename, np.array((img, emo_vector)))
            else:
                print("No face found on", img_fullpath)
                with open("faceless_ck.txt", "a+") as f:
                    f.write(str(img_fullpath)+"\n")
    else:
        print("Emotion not read well")
        print(person, seq_num)
        sys.exit()


# you can iterate glob only once
for i, f in enumerate(FILEPATHS_CK_EMOTIONS, start=0):

    # Split full path into parts seperated by /
    filename = fullpath_to_filename(f)
    person, seq_num, seq_count_text, _ = split_filename(filename)
    create_vectors(person, seq_num, seq_count_text)

    if i % 30 == 0:
        print(i, "/300")
