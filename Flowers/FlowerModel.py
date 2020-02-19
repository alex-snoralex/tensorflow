#  https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c03_exercise_flowers_with_data_augmentation.ipynb

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

import matplotlib.pyplot as plt
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D

from Flowers.ImageUtils import plot_images
from Flowers.OSUtils import organize_photos

EPOCHS = 3  # 80 is recommended
ORGANIZE_PHOTOS = False
SHOW_PRE_TRAINING_INFO = False

print("Downloading flower photos if not already present...")
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)
base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

if ORGANIZE_PHOTOS:
    organize_photos(base_dir)

print("Creating training and validation sets...")
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

batch_size = 100
IMG_SHAPE = 150

if SHOW_PRE_TRAINING_INFO:
    print("Displaying horizontal flip")
    train_image_gen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
    train_data_gen = train_image_gen.flow_from_directory(batch_size=batch_size,
                                                         directory=train_dir,
                                                         target_size=(IMG_SHAPE, IMG_SHAPE),
                                                         shuffle=True)
    augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    plot_images(augmented_images)

if SHOW_PRE_TRAINING_INFO:
    print("Displaying rotation")
    train_image_gen = ImageDataGenerator(rescale=1. / 255, rotation_range=45)
    train_data_gen = train_image_gen.flow_from_directory(batch_size=batch_size,
                                                         directory=train_dir,
                                                         target_size=(IMG_SHAPE, IMG_SHAPE),
                                                         shuffle=True)
    augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    plot_images(augmented_images)

if SHOW_PRE_TRAINING_INFO:
    print("Displaying zoom")
    train_image_gen = ImageDataGenerator(rescale=1. / 255, zoom_range=.5)
    train_data_gen = train_image_gen.flow_from_directory(batch_size=batch_size,
                                                         directory=train_dir,
                                                         target_size=(IMG_SHAPE, IMG_SHAPE),
                                                         shuffle=True)
    augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    plot_images(augmented_images)

train_image_gen = ImageDataGenerator(rescale=1. / 255,
                                     horizontal_flip=True,
                                     rotation_range=45,
                                     width_shift_range=.15,
                                     height_shift_range=.15,
                                     zoom_range=.5)
train_data_gen = train_image_gen.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     target_size=(IMG_SHAPE, IMG_SHAPE),
                                                     class_mode='sparse',
                                                     shuffle=True)
if SHOW_PRE_TRAINING_INFO:
    print("Displaying all augmentations")
    augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    plot_images(augmented_images)

val_image_gen = ImageDataGenerator(rescale=1./255)
val_data_gen = val_image_gen.flow_from_directory(batch_size=batch_size,
                                                 directory=val_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='sparse')

print("Configuring model...")
model = Sequential()

model.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training model...")
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(val_data_gen.n / float(batch_size)))
    )

print("Training complete")

acc = history.history['accuracy']  # tensorflow 1.0 = 'acc', 2.0 = 'accuracy'
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

print("\nPrinting training history...")
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./training_vs_validation.png')
