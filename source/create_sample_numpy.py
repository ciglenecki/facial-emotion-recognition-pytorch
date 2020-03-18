from __future__ import absolute_import, division, print_function, unicode_literals
import IPython.display as display
from PIL import Image  # Pillow Pil
import numpy as np
import os
import sys
from pathlib import Path, PurePath

path_project = "/home/matej/projects/fer-projekt/"
path_dataset_string = "/home/matej/projects/fer-projekt/ck+/"

path_dataset = Path(path_dataset_string)
path_emotions = Path(path_dataset, "emotions")
path_facs = Path(path_dataset, "facs")
path_images = Path(path_dataset, "images")
path_landmarks = Path(path_dataset, "landmarks")

filepaths_emotions = path_emotions.glob("*/*/*")
filepaths_facs = path_facs.glob("*/*/*")
filepaths_images = path_images.glob("*/*/*.png")
filepaths_landmarks = path_landmarks.glob("*/*/*")


def read_emotion(subject, subject_ordinal_number, sequence_count_string):
    """Reads emotion value from emotion filename
    Filename is constructed from arugments

    Args:
        subject (string)
        subject_ordinal_number (string)
        sequence_count_string (string)
    """

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


def get_image_batch_filenames(subject, subject_ordinal_number):
    """Returns image filenames for given subject and his ordinal number sequence
    Args:
        subject (string)
        subject_ordinal_number (string)
        sequence_count_string (string)
    """
    image_folder = Path(path_images, subject, subject_ordinal_number)
    image_batch_filenames = list(sorted(image_folder.glob("*.png")))

    return image_batch_filenames


def split_filename(filename):
    """Return parts of a filename
    Args:
        filename (string) - S125_001_00000014_emotion.txt
    """

    filename_parts = filename.split("_")
    subject = filename_parts[0]  # S154
    subject_ordinal_number = filename_parts[1]  # 002
    sequence_count_string = os.path.splitext(filename_parts[2])[0]
    return subject, subject_ordinal_number, sequence_count_string


def filepath_to_filename(filepath):
    filename_parts = str(filepath).split(os.path.sep)
    return filename_parts[len(filename_parts) - 1]


def create_vectors(subject, subject_ordinal_number, sequence_count_string):

    emotion = read_emotion(
        subject, subject_ordinal_number, sequence_count_string)

    if not emotion == -1:
        image_batch_filenames = get_image_batch_filenames(
            subject, subject_ordinal_number)

        # TODO: maybe throw out the `i` variable and use only split_filename
        for i, image_filename in enumerate(image_batch_filenames, start=0):

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
            p = i / (len(image_batch_filenames) - 1)
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
    else:
        print("Error")
        print(subject, subject_ordinal_number)


# you can iterate glob only once
for i, filepath_emotions in enumerate(filepaths_emotions, start=0):

    # Split full path into parts seperated by /
    filename = filepath_to_filename(filepath_emotions)
    subject, subject_ordinal_number, sequence_count_string = split_filename(
        filename)
    create_vectors(subject, subject_ordinal_number, sequence_count_string)
    if i % 30 == 0:
        print(i, "/300")
