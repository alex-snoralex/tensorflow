#  https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c03_exercise_flowers_with_data_augmentation.ipynb

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
from Flowers.ImageUtils import flower_classes, plot_images

SHOW_PRE_TRAINING_INFO = True

print("Downloading flower photos if not already present...")
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)
base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

print("Organizing training and validation sets...")
for cl in flower_classes:
    img_path = os.path.join(base_dir, cl)
    images = glob.glob(img_path + '/*.jpg')
    print("{}: {} Images".format(cl, len(images)))
    train, val = images[:round(len(images)*0.8)], images[round(len(images)*0.8):]

    for t in train:
        if not os.path.exists(os.path.join(base_dir, 'train', cl)):
            os.makedirs(os.path.join(base_dir, 'train', cl))
        shutil.move(t, os.path.join(base_dir, 'train', cl))

    for v in val:
        if not os.path.exists(os.path.join(base_dir, 'val', cl)):
            os.makedirs(os.path.join(base_dir, 'val', cl))
        shutil.move(v, os.path.join(base_dir, 'val', cl))

print("Creating training and validation sets...")
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

batch_size = 100
IMG_SHAPE = 150

if SHOW_PRE_TRAINING_INFO:
    print("Displaying horizontal flip")
    image_gen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
    train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                                   directory=train_dir,
                                                   target_size=(IMG_SHAPE, IMG_SHAPE),
                                                   shuffle=True)
    augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    plot_images(augmented_images)

if SHOW_PRE_TRAINING_INFO:
    print("Displaying rotation")
    image_gen = ImageDataGenerator(rescale=1. / 255, rotation_range=45)
    train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                                   directory=train_dir,
                                                   target_size=(IMG_SHAPE, IMG_SHAPE),
                                                   shuffle=True)
    augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    plot_images(augmented_images)

if SHOW_PRE_TRAINING_INFO:
    print("Displaying zoom")
    image_gen = ImageDataGenerator(rescale=1. / 255, zoom_range=.5)
    train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                                   directory=train_dir,
                                                   target_size=(IMG_SHAPE, IMG_SHAPE),
                                                   shuffle=True)
    augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    plot_images(augmented_images)

