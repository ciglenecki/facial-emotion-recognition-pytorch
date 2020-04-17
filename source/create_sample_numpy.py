from __future__ import absolute_import, division, print_function, unicode_literals
import IPython.display as display
from PIL import Image  # Pillow Pil
import numpy as np
import os
import sys
from pathlib import Path, PurePath

# Config
do_save_npy = True

path_project = "/home/matej/1-projects/fer-projekt/"
path_dataset_string = "/home/matej/1-projects/fer-projekt/ck+/"

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
    image_batch_filenames = list(
        sorted(image_folder.glob("*.png"), reverse=True))

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


def create_emotion_vector(i, max_sequence_number, emotion):
    p = (i-1) / (max_sequence_number-1)
    emotion_vector = [0, 0, 0, 0, 0, 0, 0]

    #emotion_vector = [0, 0, 0, 0, 0, 0, 0, 0]
    #emotion_vector[0] = round(1 - p, 3)
    emotion_vector[emotion-1] = round(p, 3)

    return emotion_vector


def create_vectors(subject, subject_ordinal_number, sequence_count_string):

    emotion = read_emotion(
        subject, subject_ordinal_number, sequence_count_string)
    if not emotion == -1:
        image_batch_filenames = get_image_batch_filenames(
            subject, subject_ordinal_number)

        """
        Find out arbitrary number of images in sequence
        Number of pictrues doesn't represent maximal needed
        1, 2, 4 => should be 4, not 3
        """
        _, _, max_sequence_number = split_filename(
            str(image_batch_filenames[0]))
        max_sequence_number = int(max_sequence_number)

        """Find image and cooresponding emotion
        Save it as .npy file
        """

        for i, image_filename in enumerate(image_batch_filenames, start=0):

            _, _, i = split_filename(str(image_filename))
            i = int(i)

            image_fullpath = Path(
                path_images, subject, subject_ordinal_number, image_filename
            )

            image = np.asarray(Image.open(image_fullpath).convert("L"))

            emotion_vector = create_emotion_vector(
                i, max_sequence_number, emotion)

            emotion_vector = np.array(emotion_vector)
            print(emotion_vector)
            if sum(emotion_vector) > 1:
                print("Emotion_vector: ", emotion_vector)
                print("Error sum of emotion for file ", image_filename)
                sys.exit()

            # Save image
            if (do_save_npy):
                np.save(
                    path_project
                    + "numpy/"
                    + subject
                    + "_"
                    + subject_ordinal_number
                    + "_"
                    + str(i).zfill(8),
                    np.array((image, emotion_vector)))
    else:
        print("Error; emotion is -1")
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
