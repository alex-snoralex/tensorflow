import tensorflow as tf


def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels
