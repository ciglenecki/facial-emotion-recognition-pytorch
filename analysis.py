from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import IPython.display as display
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pathlib

# ck+
#     ├── emotions
#     │   ├── S005
#     │   │   └── 001
#     │   │   │   ├── S005_001_00000011_emotion.txt
#     │   │   │   │   └── 3.0000000e+00
#     │   ├── S010
#     │   │   ├── 001
#     │   │   │   EMPTY!
#     │   │   ├── 002
#     │   │   │   ├── S010_002_00000014
# _emotion.txt
#     │   │   │   │   └── 7.0000000e+00
#     │   │   ├── 003
#             ...

#     ├── facs
#             ├── S005
#             ├── S010
#             ├── S011
#             ...
#     ├── images
#             ├── S005
#             ├── S010
#             ├── S011
#             ...
#     └── landmarks
#             ├── S005
#             ├── S010
#             ├── S011
#             ...

# IMG_WIDTH = 640
# IMG_HEIGHT = 490

emotion_declaration = {
 'anger': 0,
 'disgust': 1,
 'fear': 2,
 'happiness': 3,
 'sadness': 4,
 'surprise': 5,
 'calm': 6
}

AUTOTUNE = tf.data.experimental.AUTOTUNE

dataset_location = '/run/media/matej/hdd-main/1-projects/1-tech/2019-projekt/ck+'

data_dir = pathlib.Path(dataset_location)
ds = list()

emotions = tf.data.Dataset.list_files(str(data_dir/'emotions/*/*/*'))
facs = tf.data.Dataset.list_files(str(data_dir/'facs/*/*/*'))
images = tf.data.Dataset.list_files(str(data_dir/'images/*/*/*'))
landmarks = tf.data.Dataset.list_files(str(data_dir/'landmarks/*/*/*'))

data_files = tf.data.Dataset.list_files(str(data_dir/'*/*/*/*'))

for file_emotion in emotions.take(5):
    print(file_emotion)
    print(os.path.splitext(file_emotion.numpy())[0])

for file_emotion in emotions.take(5):
    parts = tf.strings.split(file_emotion, os.path.sep)
    file_name = parts[len(parts) - 1]
    print(file_name)

data_directories = np.array([item.name for item in data_dir.glob('*')])


def process_path(file_path):
    print(file_path)
    parts = tf.strings.split(file_path, os.path.sep)

    os.path.splitext(file_emotion.numpy())[0]
    file_name = parts[len(parts) - 1].numpy()

    print(file_name)
    return file_name


# num_parallel_calls = multiple images are processed in parallel
labeled_ds = emotions.map(process_path, num_parallel_calls=AUTOTUNE)
print(labeled_ds)

# def decode_img(img):
#   # string -> 3D uint8 tensor
#   img = tf.image.decode_jpeg(img, channels=3)
#   # floats -> [0, 1].
#   img = tf.image.convert_image_dtype(img, tf.float32)
#   # all pictures, same size
#   return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
