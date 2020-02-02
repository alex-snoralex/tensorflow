import tensorflow as tf
import tensorflow_datasets as tfds
import math
import numpy as np
import matplotlib.pyplot as plt
import logging

tf.compat.v1.enable_eager_execution()
tfds.disable_progress_bar()

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True, shuffle_files=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))


def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


# The map function applies the normalize function to each element in the train
# and test datasets
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# The first time you use the dataset, the images will be loaded from disk
# Caching will keep them in memory, making training faster
train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()

# ~~~~~~~~ Good above ~~~~~~~~~~

# Take a single image, and remove the color dimension by reshaping
# img = 0
# for image, label in test_dataset.take(1):
#     img = image.numpy().reshape((28, 28))
#     break
img = test_dataset.take(1)
print("cool stuff incoming...")
print(img.numpy())
img = img.numpy().reshape((28, 28))

# Plot the image - voila a piece of fashion clothing
plt.figure()
# plt.imshow(img, cmap=plt.cm)
plt.imshow(img, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()
