from __future__ import absolute_import, division, print_function, unicode_literals
import IPython.display as display
from PIL import Image  # Pillow Pil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path, PurePath
from tempfile import TemporaryFile


# ck+
#     |----- emotions
#     |   |----- S005
#     |   |   ----- 001
#     |   |   |   |----- S005_001_00000011_emotion.txt
#     |   |   |   |   ----- 3.0000000e+00
#     |   |----- S010
#     |   |   |----- 001
#     |   |   |   EMPTY!
#     |   |   |----- 002
#     |   |   |   |----- S010_002_00000014_emotion.txt
#     |   |   |   |   ----- 7.0000000e+00
#     |   |   |----- 003
#             ...

#     |----- facs
#             |----- S005
#             |----- S010
#             |----- S011
#             ...
#     |----- images
#             |----- S005
#             |----- S010
#             |----- S011
#             ...
#     ----- landmarks
#             |----- S005
#             |----- S010
#             |----- S011
#             ...


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

# AUTOTUNE = tf.data.experimental.AUTOTUNE

IMG_WIDTH = 640
IMG_HEIGHT = 490

path_project = "/home/matej/projects/fer-projekt/"
path_dataset_string = "/home/matej/projects/fer-projekt/ck+/"

path_dataset = Path(path_dataset_string)
path_emotions = Path(path_dataset, "emotions")
path_facs = Path(path_dataset, "facs")
path_images = Path(path_dataset, "images")
path_landmarks = Path(path_dataset, "landmarks")

# filepaths_emotions = tf.data.Dataset.list_files(str(str(path_emotions) + '*/*/*'))
filepaths_emotions = path_emotions.glob("*/*/*")
filepaths_facs = path_facs.glob("*/*/*")
filepaths_images = path_images.glob("*/*/*.png")
filepaths_landmarks = path_landmarks.glob("*/*/*")

print("++++++++\n++++++++\n++++++++\n")


def get_emotion(subject, subject_ordinal_number, sequence_count_string):

    # By given arguments function constructs a filename
    # Finds the file
    # Reads it's emotion value and returns it
    emotion_filename = "_".join(
        [subject, subject_ordinal_number, sequence_count_string, "emotion.txt"]
    )
    emotion_fullpath = Path(
        path_emotions, subject, subject_ordinal_number, emotion_filename
    )
    if os.path.isfile(emotion_fullpath):
        emotion_file = open(str(emotion_fullpath), "r",)
    else:
        return -1
    emotion = int(float(emotion_file.readline()))
    return emotion


# Todo: rewrite to get_image
def get_image_filenames(subject, subject_ordinal_number, sequence_count_string):

    # By given arguments function constructs a filename
    # Finds the file
    # Reads all images and returns them
    image_folder = Path(path_images, subject, subject_ordinal_number)
    image_filenames = list(sorted(image_folder.glob("*.png")))

    return image_filenames


def split_filename(filename):
    # filename = S125_001_00000014_emotion.txt
    filename_parts = filename.split("_")
    subject = filename_parts[0]  # S154
    subject_ordinal_number = filename_parts[1]  # 002
    sequence_count_string = os.path.splitext(filename_parts[2])[0]
    return subject, subject_ordinal_number, sequence_count_string


def encode_emotion(emotion):
    # return tf.convert_to_tensor(np.array(emotion))
    pass


def decode_image(img):
    # convert the compressed string to a 3D uint8 tensor
    # img = tf.convert_to_tensor(img, dtype=tf.uint8)
    # resize the image to the desired size.
    # return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    pass


def create_vectors(subject, subject_ordinal_number, sequence_count_string):
    print(subject)
    emotion = get_emotion(subject, subject_ordinal_number,
                          sequence_count_string)
    if not emotion == -1:
        image_filenames = get_image_filenames(
            subject, subject_ordinal_number, sequence_count_string
        )
        for i, image_filename in enumerate(image_filenames, start=0):

            # Fixing if `i` value if needed
            _, _, sequence_count_check = split_filename(str(image_filename))
            sequence_count_check = int(sequence_count_check)
            if not i+1 == (sequence_count_check):
                i = sequence_count_check

            # Construct image from image path
            image_fullpath = Path(
                path_images, subject, subject_ordinal_number, image_filename
            )
            image = np.asarray(Image.open(image_fullpath).convert("L"))

            # Find out emotion level
            p = i / (len(image_filenames) - 1)
            emotion_current = [0, 0, 0, 0, 0, 0, 0, 0]
            emotion_current[0] = round(1 - p, 3)
            emotion_current[emotion] = round(p, 3)
            emotion_current = np.array(emotion_current)

            # Save image
            np.save(
                path_project
                + "numpy/"
                + subject
                + "_"
                + subject_ordinal_number
                + "_"
                + str(i).zfill(8),
                np.array((image, emotion_current)))

            # Save to datastrcture
            # vectors.append([image, emotion_current])
    else:
        print(subject, subject_ordinal_number)


vectors = []

# you can iterate glob only once
for filepath_emotions in filepaths_emotions:

    # Split full path into parts seperated by /
    filename_parts = str(filepath_emotions).split(os.path.sep)
    filename = filename_parts[len(filename_parts) - 1]
    subject, subject_ordinal_number, sequence_count_string = split_filename(
        filename)
    create_vectors(subject, subject_ordinal_number, sequence_count_string)
