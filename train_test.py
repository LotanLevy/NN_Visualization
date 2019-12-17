from __future__ import absolute_import, division, print_function
import tensorflow as tf

import numpy as np
from utils import tensor_to_numpy

class ImageTrainer:
    """
    Manage the train step
    """
    def __init__(self, target_neuron, optimizer, image, min_value_target, regression_factor=0.2):
        self.target_neuron = target_neuron
        self.optimizer = optimizer
        self.min_value_target = tf.constant(min_value_target, dtype=tf.float32)
        self.regression_factor = np.float32(regression_factor)
        self.end_training = False
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')



    def get_step(self):

        @tf.function()
        def train_step(image):
            with tf.GradientTape() as tape:
                prediction = self.target_neuron(image)
                loss = self.calculate_loss(prediction, image)

            gradients = tape.gradient(loss, image)
            self.optimizer.apply_gradients([(gradients, image)])
            self.train_loss(loss)

        return train_step

    def calculate_loss(self, prediction, image):
        square_norm = tf.math.reduce_mean(tf.math.square(tf.math.sqrt(tf.math.square(image))))
        loss = -(prediction - self.regression_factor * square_norm)
        return loss





