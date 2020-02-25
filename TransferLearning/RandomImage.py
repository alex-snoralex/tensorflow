# https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l06c01_tensorflow_hub_and_transfer_learning.ipynb
# This class will download an image from a given url and, using tfhub's mobilenet, predict what's in the image

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pylab as plt
import numpy as np
import PIL.Image as Image
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# tf.keras.utils.get_file() will try to download from the url given unless the name given is already cached.
# Note that random urls off the internet might return 403 forbidden errors.
# EXAMPLE_IMAGE_URL = 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg'
TEST_IMAGE_NAME = 'Jokes1.jpg'
TEST_IMAGE_URL = 'https://vignette.wikia.nocookie.net/animals-are-cool/images/5/5e/Lunge_popup.jpg/revision/latest?cb' \
                 '=20180313213817 '

CLASSIFIER_URL = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4'
CLASSIFIER_LABELS = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'

IMAGE_RES = 224

model = tf.keras.Sequential([
    hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
])

test_image = tf.keras.utils.get_file(TEST_IMAGE_NAME, TEST_IMAGE_URL)
test_image = Image.open(test_image).resize((IMAGE_RES, IMAGE_RES))
test_image = np.array(test_image) / 255.0
print("Test image shape: ", test_image.shape)

result = model.predict(test_image[np.newaxis, ...])
print("Result shape:", result.shape)

predicted_label_number = np.argmax(result[0], axis=-1)
print("Predicted label number: ", predicted_label_number)

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', CLASSIFIER_LABELS)
image_net_labels = np.array(open(labels_path).read().splitlines())
predicted_label_name = image_net_labels[predicted_label_number]
plt.imshow(test_image)
plt.axis('off')
_ = plt.title("Prediction: " + predicted_label_name.title())
print("Prediction: ", predicted_label_name.title())
print("Showing prediction and image...")
plt.show()
