from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

dataset_location = '/run/media/matej/hdd-main/1-projects/1-tech/2019-projekt/ck+'

data_dir = pathlib.Path(dataset_location)

list_ds = tf.data.Dataset.list_files(str(data_dir, '/*/'))
# CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])

# IMG_WIDTH = 640
# IMG_HEIGHT = 490

# for f in list_ds.take(5):
#   print(f.numpy())

# def get_label(file_path):
#   parts = tf.strings.split(file_path, os.path.sep)
#   # Second to last is classnames
#   return parts[-2] == CLASS_NAMES


# def decode_img(img):
#   # string -> 3D uint8 tensor
#   img = tf.image.decode_jpeg(img, channels=3)
#   # floats -> [0, 1].
#   img = tf.image.convert_image_dtype(img, tf.float32)
#   # all pictures, same size
#   return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])