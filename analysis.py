from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import IPython.display as display
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path, PurePath
import cv2

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
#     │   │   │   ├── S010_002_00000014_emotion.txt
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



emotion_declaration = [
    "neutral",
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "sadness",
    "surprise"
]

AUTOTUNE = tf.data.experimental.AUTOTUNE

IMG_WIDTH = 640
IMG_HEIGHT = 490



path_dataset_string = '/home/matej/projects/fer-projekt/ck+/'

path_dataset = Path(path_dataset_string)
path_emotions = Path(path_dataset , 'emotions')
path_facs = Path(path_dataset , 'facs')
path_images = Path(path_dataset , 'images')
path_landmarks = Path(path_dataset , 'landmarks')

# filepaths_emotions = tf.data.Dataset.list_files(str(str(path_emotions) + '*/*/*'))
filepaths_emotions = tf.data.Dataset.list_files(str(path_emotions/'*/*/*'))
filepaths_facs= tf.data.Dataset.list_files(str(path_facs/'*/*/*'))
filepaths_images = tf.data.Dataset.list_files(str(path_images/'*/*/*'))
filepaths_landmarks = tf.data.Dataset.list_files(str(path_landmarks/'*/*/*'))

filepaths_all = tf.data.Dataset.list_files(str(path_dataset /'*/*/*/*'))
filepaths_directories = np.array([item.name for item in path_dataset .glob('*')])

for filepath_emotions in filepaths_emotions.take(5):
    print(filepath_emotions)
    print(os.path.splitext(filepath_emotions.numpy())[0])

for filepath_emotions in filepaths_emotions.take(5):
    parts = tf.strings.split(filepath_emotions, os.path.sep)
    filename = parts[len(parts) - 1]
    print(filename)

print("++++++++\n++++++++\n++++++++\n")

def get_emotion(subject, subject_ordinal_number, sequence_count_string):

    # By given arguments function constructs a filename
    # Finds the file
    # Reads it's emotion value and returns it
    emotion_filename = '_'.join([subject, subject_ordinal_number, sequence_count_string, 'emotion.txt'])
    emotion_file  = open(str(Path(path_emotions, subject, subject_ordinal_number, emotion_filename)), 'r')
    emotion = int(float(emotion_file.readline()))
    return emotion

def get_images(subject, subject_ordinal_number, sequence_count_string):

    # By given arguments function constructs a filename
    # Finds the file
    # Reads all images and returns them

    images = []
    image_folder  = Path(path_images, subject, subject_ordinal_number)
    image_filenames = list(image_folder.glob('*.png'))    

    # Read each image filename and add image to images[]
    for image_filename in image_filenames:            
        image_fullpath  = Path(path_images, subject, subject_ordinal_number, image_filename)
        images.append(np.array(Image.open(image_fullpath).convert('LA')))
    return images

def process_filepath(filepath):
    filepath_parts = filepath.split('_')
    subject = filepath_parts[0] #S154
    subject_ordinal_number = filepath_parts[1] #002
    sequence_count_string = filepath_parts[2]

    return subject, subject_ordinal_number, sequence_count_string

def encode_emotion(emotion):
    return tf.convert_to_tensor(np.array(emotion))

def decode_image(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.convert_to_tensor(img, dtype=tf.uint8)    
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def create_dataset(filepath):

    # define properties
    subject, subject_ordinal_number, sequence_count_string = process_filepath(filepath)
    emotions = []
    # find values
    emotion = get_emotion(subject, subject_ordinal_number, sequence_count_string)
    images = get_images(subject, subject_ordinal_number, sequence_count_string)
    sequence_count_override = len(images)
    dataset = []

    for i in range(0,len(images)):
        # 0. index neutral i index EMOCIJE 
        p = i/(len(images) - 1)
        emotion_current = [0,0,0,0,0,0,0,0]
        emotion_current[0] = round(1 - p, 3)
        print(emotion)
        emotion_current[emotion] = round(p, 3)
        # TODO: after image decoding save 1 label and 1 image to TF dataset
        decoded_image = decode_image(images[i])        
        dataset.append([decoded_image, encode_emotion(emotion_current)])
    for i in dataset:
        print(i)
    dataset_tensor = tf.data.Dataset.from_tensor_slices(dataset)
    return dataset_tensor
    
    
for filepath_emotions in filepaths_emotions:
    # Split full path into parts seperated by /    
    parts = tf.strings.split(filepath_emotions, os.path.sep)
    filepath = (parts[len(parts) - 1]).numpy().decode('UTF-8')
    image, emotion = create_dataset(filepath)

    
# result = tf.map_fn(process_path, filepaths_emotions)
# print(result)

# def decode_img(img):
#   # string -> 3D uint8 tensor
#   img = tf.image.decode_jpeg(img, channels=3)
#   # floats -> [0, 1].
#   img = tf.image.convert_image_dtype(img, tf.float32)
#   # all pictures, same size
#   return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
