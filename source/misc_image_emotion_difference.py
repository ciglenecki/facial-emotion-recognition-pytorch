from __future__ import absolute_import, division, print_function, unicode_literals
import IPython.display as display
from PIL import Image  # Pillow Pil
import numpy as np
import os
import sys
from pathlib import Path, PurePath

# Config
SAVE_NPY = True
NEUTRAL_EXISTS = True

# Paths
PATH_PROJECT = Path.cwd()
PATH_NUMPY = Path(PATH_PROJECT, "numpy")


PATH_DATASET_STRING = Path(PATH_PROJECT, "ck+")

PATH_DATASET = Path(PATH_DATASET_STRING)
PATH_EMOTIONS = Path(PATH_DATASET, "emotions")
PATH_FACS = Path(PATH_DATASET, "facs")
PATH_IMAGES = Path(PATH_DATASET, "images")
PATH_LANDMARKS = Path(PATH_DATASET, "landmarks")

FILEPATHS_EMOTIONS = PATH_EMOTIONS.glob("*/*/*")
FILEPATHS_FACS = PATH_FACS.glob("*/*/*")
FILEPATHS_IMAGES = PATH_IMAGES.glob("*/*/*.png")
FILEPATHS_LANDMARKS = PATH_LANDMARKS.glob("*/*/*")

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
        pre_path = PATH_EMOTIONS
    elif(file_type_name == "image"):
        pre_path = PATH_IMAGES

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


total = 0
total_emo = 0
total_img = 0

for i, f in enumerate(FILEPATHS_EMOTIONS, start=0):

    filename = fullpath_to_filename(f)
    person, seq_num, seq_count_text, _ = split_filename(filename)
    img_folder = attr_to_prefolder(person, seq_num, "image")
    img_batch_filenames = list(img_folder.glob("*.png"))

    emo = int(seq_count_text)
    img = len(img_batch_filenames)

    # if emo != img:
    #     print(img_folder)

    total_emo = total_emo + emo
    total_img = total_img + img
    total = total + (emo - img)

print("Images: ", total_img)
print("Emotions: ", total_emo)
print("Difference", total)
