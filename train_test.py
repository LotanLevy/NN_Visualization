from __future__ import absolute_import, division, print_function
import tensorflow as tf

import numpy as np
from utils import tensor_to_numpy

class ImageTrainer:
    """
    Manage the train step
    """
    def __init__(self, target_neuron, optimizer, loss_name, regression_factor):
        self.target_neuron = target_neuron
        self.optimizer = optimizer
        self.regression_factor = np.float32(regression_factor)
        self.end_training = False
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.last_pred = tf.keras.metrics.Mean(name='last_pred')
        self.loss_name = loss_name




    def get_step(self):

        @tf.function()
        def train_step(image):
            with tf.GradientTape() as tape:
                prediction = self.target_neuron.get_relevant_layer()(image)
                for idx in self.target_neuron.neuron_indices:
                    prediction = prediction[idx]
                loss = self.calculate_loss(prediction, image)

            gradients = tape.gradient(loss, image)
            self.optimizer.apply_gradients([(gradients, image)])
            self.train_loss(loss)

        return train_step

    def calculate_loss(self, prediction, image):
        if self.loss_name == "Fourier":
            fft_image = tf.signal.fft(tf.cast(image, tf.complex64))
            w = tf.cast(tf.linalg.norm(fft_image), tf.float32)
            reg = tf.cast(tf.reduce_mean(tf.abs(image) - 1/w), tf.float32)
        else:
            reg = self.regression_factor * tf.reduce_mean(tf.square(tf.sqrt(tf.square(image))))
        loss = -(prediction - reg)
        self.last_pred.reset_states()
        self.last_pred(prediction)
        return loss





