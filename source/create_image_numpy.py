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
    image_filenames = list(image_folder.glob("*.png"))

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

    emotion = get_emotion(subject, subject_ordinal_number,
                          sequence_count_string)
    if not emotion == -1:
        image_filenames = get_image_filenames(
            subject, subject_ordinal_number, sequence_count_string
        )
        for i, image_filename in enumerate(image_filenames, start=0):

            image_fullpath = Path(
                path_images, subject, subject_ordinal_number, image_filename
            )
            image = np.asarray(Image.open(image_fullpath).convert("L"))
            # 0. index neutral i index EMOCIJE
            p = i / (len(image_filenames) - 1)
            emotion_current = [0, 0, 0, 0, 0, 0, 0, 0]
            emotion_current[0] = round(1 - p, 3)
            emotion_current[emotion] = round(p, 3)
            emotion_current = np.array(emotion_current, dtype=np.uint8)
            # TODO: after image decoding save 1 label and 1 image to TF dataset
            # decoded_image = decode_image(images[i]) if doEncode else images[i]
            # encoded_emotion = (
            #     encode_emotion(emotion_current) if doEncode else emotion_current
            # )
            print(emotion_current.dtype)
            np.save(
                path_project
                + "numpy/"
                + subject
                + "_"
                + subject_ordinal_number
                + "_"
                + str(i).zfill(8),
                np.array([image, emotion_current], dtype=[
                         ('image', 'u1'), ('emotion', 'u1')])

            )
            vectors.append([image, emotion_current])
    else:
        print(subject, subject_ordinal_number)


vectors = []

# you can iterate glob only once
for filepath_emotions in filepaths_emotions:
    # Split full path into parts seperated by /
    parts = str(filepath_emotions).split(os.path.sep)
    filename = parts[len(parts) - 1]
    subject, subject_ordinal_number, sequence_count_string = split_filename(
        filename)
    create_vectors(subject, subject_ordinal_number, sequence_count_string)

print(vectors)
# image, emotion = create_dataset(filepath)

# result = tf.map_fn(process_path, filepaths_emotions)
# print(result)

# def decode_img(img):
#   # string -> 3D uint8 tensor
#   img = tf.image.decode_jpeg(img, channels=3)
#   # floats -> [0, 1].
#   img = tf.image.convert_image_dtype(img, tf.float32)
#   # all pictures, same size
#   return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


# Loading files
# filepaths_emotions = tf.data.Dataset.list_files(str(path_emotions/'*/*/*'))

#    print(os.path.splitext(filepath_emotions.numpy())[0])
