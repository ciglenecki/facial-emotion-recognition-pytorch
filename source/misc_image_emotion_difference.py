from __future__ import absolute_import, division, print_function, unicode_literals
import IPython.display as display
from PIL import Image  # Pillow Pil
import numpy as np
import os
import sys
from pathlib import Path, PurePath
from paths import *
NEUTRAL_EXISTS = True


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


def attr_to_prefolder(person, seq_num, file_type_name=None):

    if(file_type_name == "emotion"):
        pre_path = PATH_CK_EMOTIONS
    elif(file_type_name == "image"):
        pre_path = PATH_CK_IMAGES

    return Path(pre_path, person, seq_num)


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


total = 0
total_emo = 0
total_img = 0

for i, f in enumerate(FILEPATHS_CK_EMOTIONS, start=0):

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
