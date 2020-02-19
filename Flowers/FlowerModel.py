#  https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c03_exercise_flowers_with_data_augmentation.ipynb

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
from Flowers.ImageUtils import flower_classes, plot_images

EPOCHS = 3  # 80 is recommended
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
                                     zoom_range=.5)
train_data_gen = train_image_gen.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     target_size=(IMG_SHAPE, IMG_SHAPE),
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
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("Training model...")
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.__sizeof__() / float(batch_size))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(val_data_gen.__sizeof__() / float(batch_size)))
    )

print("Training complete")

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