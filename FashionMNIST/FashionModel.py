# https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l03c01_classifying_images_of_clothing.ipynb

import tensorflow as tf
import tensorflow_datasets as tfds
import math
import numpy as np
import matplotlib.pyplot as plt
import logging

from FashionMNIST.DataUtils import normalize
from FashionMNIST.ImagePlotUtils import plot_image, plot_value_array, class_names, show_plot

NUMBER_OF_EPOCHS = 5
SHOW_TRAINING_IMAGES = False
SHOW_PREDICTION_IMAGES = True

tf.compat.v1.enable_eager_execution()
tfds.disable_progress_bar()

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print("Downloading Tensorflow dataset if not already present...")
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True, shuffle_files=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))

# The map function applies the normalize function to each element in the train and test datasets
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# The first time you use the dataset, the images will be loaded from disk
# Caching will keep them in memory, making training faster
train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()

# Take a single image, and remove the color dimension by reshaping
img = 0
for image, label in test_dataset.take(1):
    img = image.numpy().reshape((28, 28))
    break

plt.figure()
plt.imshow(img, cmap=plt.cm.get_cmap("binary"))
plt.colorbar()
plt.grid(False)
show_plot(SHOW_TRAINING_IMAGES, "Plotting single image")

plt.figure(figsize=(10, 10))
i = 0
for (image, label) in test_dataset.take(25):
    image = image.numpy().reshape((28, 28))
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.get_cmap("binary"))
    plt.xlabel(class_names[label])
    i += 1
show_plot(SHOW_TRAINING_IMAGES, "Plotting grid of images")

# The actual ML part of this class
print("Building the model...")
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training the model...")
BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
model.fit(train_dataset, epochs=NUMBER_OF_EPOCHS, steps_per_epoch=math.ceil(num_train_examples / BATCH_SIZE))

print("Testing the model...")
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples / 32))
print('Accuracy on test dataset:', test_accuracy)

predictions, test_images, test_labels = 0, 0, 0
for t_images, t_labels in test_dataset.take(1):
    test_images = t_images.numpy()
    test_labels = t_labels.numpy()
    predictions = model.predict(t_images)

i = 2
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
show_plot(SHOW_PREDICTION_IMAGES, "Test image #{} and prediction".format(i))

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
show_plot(SHOW_PREDICTION_IMAGES, "Test image #{} and prediction".format(i))

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
show_plot(SHOW_PREDICTION_IMAGES, "Plotting grid of test images and predictions")

# Grab an image from the test dataset
img = test_images[0]
print("Image shape from dataset: {}".format(img.shape))

# Add the image to a batch where it's the only member.
img = np.array([img])
print("Image shape by itself : {}".format(img.shape))

predictions_single = model.predict(img)
print("Image prediction: {}".format(predictions_single))

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
show_plot(SHOW_PREDICTION_IMAGES, "Plotting bar chart of predictions for first image.")
