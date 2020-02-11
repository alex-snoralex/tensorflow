# https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c01_dogs_vs_cats_without_augmentation.ipynb
# Super CatDog model incorporates dropout and image augmentation to make model more resilient.

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

from CatsAndDogs.ImageUtils import plot_images

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

SHOW_PRE_TRAINING_INFO = False
SHOW_AUGMENTATIONS = False
SHOW_POST_TRAINING_INFO = False
EPOCHS = 5  # 100 is recommended

print("Downloading 'Cats & Dogs' dataset if not already present...")
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)

base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

print("\nConfiguring training and validation datasets...")
train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

if SHOW_PRE_TRAINING_INFO:
    print('total training cat images:', num_cats_tr)
    print('total training dog images:', num_dogs_tr)

    print('total validation cat images:', num_cats_val)
    print('total validation dog images:', num_dogs_val)
    print("--")
    print("Total training images:", total_train)
    print("Total validation images:", total_val)

BATCH_SIZE = 100  # Number of training examples to process before updating our models variables
IMG_SHAPE = 150  # Our training data consists of images with width of 150 pixels and height of 150 pixels

print("Augmenting images...")

if SHOW_AUGMENTATIONS:
    print("Displaying horizontal flip")
    image_gen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
    train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                                   directory=train_dir,
                                                   shuffle=True,
                                                   target_size=(IMG_SHAPE, IMG_SHAPE))

    augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    plot_images(augmented_images)

if SHOW_AUGMENTATIONS:
    print("Displaying image rotation")
    image_gen = ImageDataGenerator(rescale=1. / 255, rotation_range=45)
    train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                                   directory=train_dir,
                                                   shuffle=True,
                                                   target_size=(IMG_SHAPE, IMG_SHAPE))
    augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    plot_images(augmented_images)

if SHOW_AUGMENTATIONS:
    print("Displaying zoom")
    image_gen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.5)
    train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                                   directory=train_dir,
                                                   shuffle=True,
                                                   target_size=(IMG_SHAPE, IMG_SHAPE))
    augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    plot_images(augmented_images)

print("Applying augmentations to training set...")
image_gen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_SHAPE, IMG_SHAPE))
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
if SHOW_AUGMENTATIONS:
    plot_images(augmented_images)

print("Creating validation set...")
validation_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our validation data
val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              shuffle=False,
                                                              target_size=(IMG_SHAPE, IMG_SHAPE),  # (150,150)
                                                              class_mode='binary')

if SHOW_PRE_TRAINING_INFO:
    print("Plotting a small batch of training images...")
    sample_training_images, _ = next(train_data_gen)
    plot_images(sample_training_images[:5])  # Plot images 0-4

print("\nConfiguring the model...")
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

if SHOW_PRE_TRAINING_INFO:
    print(model.summary())

print("\nTraining the model...")
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)
print("Training complete")

# print("Here are the history keys: {}", history.history.keys())
acc = history.history['acc']  # tensorflow 1.0 = 'acc', 2.0 = 'accuracy'
val_acc = history.history['val_acc']
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
plt.savefig('./super_training_vs_validation.png')

if SHOW_POST_TRAINING_INFO:
    plt.show()
