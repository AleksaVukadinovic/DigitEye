import os
import numpy as np
import cv2 as cv
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) =mnist.load_data()

X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
