import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pylab as plt
import logging
from TransferLearning.ImageUtil import format_image

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

split = [tfds.Split.TRAIN.subsplit(tfds.percent[:70]),
         tfds.Split.TRAIN.subsplit(tfds.percent[70:])]

print("Loading flowers dataset...")
(training_set, validation_set), dataset_info = tfds.load('tf_flowers',
                                                         split=split,
                                                         with_info=True,
                                                         as_supervised=True)
print("Finished loading flowers dataset.")


num_total_examples = dataset_info.splits['train'].num_examples
num_training_examples = num_total_examples * 0.7
num_validation_examples = num_total_examples * 0.3
num_classes = dataset_info.features['label'].num_classes

print('Total Number of Classes: {}'.format(num_classes))
print('Total Number of Training Images: {}'.format(num_training_examples))
print('Total Number of Validation Images: {} \n'.format(num_validation_examples))