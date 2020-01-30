from Celsius2Fahrenheit.Formula import celsius_2_fahrenheit

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

celsius_features = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_labels = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)
celsius_test_set = np.array([-56.2, -22.22, -3.1, 12.12, 45, 102, 453], dtype=float)

print("\nEntering training set: ")
for i, c in enumerate(celsius_features):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_labels[i]))

print("\nTraining the model")
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius_features, fahrenheit_labels, epochs=1000, verbose=False)
print("Finished training the model\n")

print("Beginning test set: ")
for i, c in enumerate(celsius_test_set):
    predicted = round(float(model.predict([celsius_test_set[i]])), 2)
    print("{} degrees Celsius. {} Expected Fahrenheit. {} Predicted Fahrenheit."
          .format(celsius_test_set[i], celsius_2_fahrenheit(c), predicted))

print("Layer variables: {}".format(l0.get_weights()))

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()
