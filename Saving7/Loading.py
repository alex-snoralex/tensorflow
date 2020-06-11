import time
import numpy as np
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

from tensorflow.keras import layers

# Stuff from Saving section
export_path_keras = "./model_1591852482.h5"
(train_examples, validation_examples), info = tfds.load(
    'cats_vs_dogs',
    split=[tfds.Split.TRAIN.subsplit(tfds.percent[:80]),
           tfds.Split.TRAIN.subsplit(tfds.percent[80:])],
    with_info=True,
    as_supervised=True,
)


def format_image(image, label):
    # `hub` image modules expect their data normalized to the [0,1] range.
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES)) / 255.0
    return image, label


num_examples = info.splits['train'].num_examples
BATCH_SIZE = 32
IMAGE_RES = 224
train_batches = train_examples.cache().shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)
image_batch, label_batch = next(iter(train_batches.take(1)))

# Part 4: Loading the Keras .h5 model file, make more predictions
reloaded = tf.keras.models.load_model(
    export_path_keras,
    # `custom_objects` tells keras how to load a `hub.KerasLayer`
    custom_objects={'KerasLayer': hub.KerasLayer})

reloaded.summary()

reloaded_result_batch = reloaded.predict(image_batch)
reloaded_result_batch = tf.squeeze(reloaded_result_batch).numpy()
predicted_ids = np.argmax(reloaded_result_batch, axis=-1)
class_names = np.array(info.features['label'].names)
predicted_class_names = class_names[predicted_ids]
print("Predicted class names: ", predicted_class_names)

print("Labels: ", label_batch)
print("Predicted labels: ", predicted_ids)

# Part 4.5: Keep training model
# EPOCHS = 3
# history = reloaded.fit(train_batches,
#                        epochs=EPOCHS,
#                        validation_data=validation_batches)

# Part 5: Export as TensorFlow Saved Model
t = time.time()
export_path_sm = "./tfsm_{}".format(int(t))
print(export_path_sm)
tf.saved_model.save(reloaded, export_path_sm)

# Part 6: Load SavedModel
# reloaded_sm = tf.saved_model.load(export_path_sm)
# reload_sm_result_batch = reloaded_sm(image_batch, training=False).numpy()

# Part 7: Loading the SavedModel as a Keras Model
# t = time.time()
# export_path_sm = "./{}".format(int(t))
# print(export_path_sm)
# tf.saved_model.save(reloaded, export_path_sm)
# reload_sm_keras = tf.keras.models.load_model(
#   export_path_sm,
#   custom_objects={'KerasLayer': hub.KerasLayer})
# reload_sm_keras.summary()
# reload_sm_keras_result_batch = reload_sm_keras.predict(image_batch)
