from Celsius2Fahrenheit.Formula import celsius_2_fahrenheit

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

celsius_q = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)
celsius_test = 100.00

print("\nEntering test set: ")
for i, c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model\n")

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

predicted = round(float(model.predict([celsius_test])[0][0]), 2)
print("For a celsius value of {}, the model predicted {} when the actual is {}."
      .format(celsius_test, predicted, celsius_2_fahrenheit(celsius_test)))
print("Layer variables: {}".format(l0.get_weights()))
