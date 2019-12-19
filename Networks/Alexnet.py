from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NN import NN

import tensorflow as tf

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Activation, MaxPooling2D, Dropout
# from tensorflow.keras import Model


class Alexnet(NN):
    def __init__(self):
        super(Alexnet, self).__init__()
        # OPS
        self.relu = Activation('relu')
        self.maxpool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')
        # self.dropout = Dropout(0.4)
        self.softmax = Activation('softmax')

        # Conv layers
        self.conv1 = Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11), strides=(4, 4), padding='same', trainable=False)
        self.conv2a = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', trainable=False)
        self.conv2b = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', trainable=False)
        self.conv3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', trainable=False)
        self.conv4a = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', trainable=False)
        self.conv4b = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', trainable=False)
        self.conv5a = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', trainable=False)
        self.conv5b = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', trainable=False)

        # Fully-connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(4096, input_shape=(100,), trainable=False)
        self.dense2 = Dense(4096, trainable=False)
        self.dense3 = Dense(1000, trainable=False)

        self.norm1 = lambda x: tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
        self.concat2 = lambda x: tf.concat((self.conv2a(x[:, :, :, :48]), self.conv2b(x[:, :, :, 48:])), 3)
        self.concat4 = lambda x: tf.concat((self.conv4a(x[:, :, :, :192]), self.conv4b(x[:, :, :, 192:])), 3)
        self.concat5 = lambda x: tf.concat((self.conv5a(x[:, :, :, :192]), self.conv5b(x[:, :, :, 192:])), 3)



        self.all_layers = [self.conv1, self.relu, self.norm1, self.maxpool, self.concat2, self.relu, self.norm1, self.maxpool,
                           self.conv3, self.relu, self.concat4, self.relu, self.concat5, self.relu, self.maxpool,
                           self.flatten, self.dense1, self.relu, self.dense2, self.relu, self.dense3, self.softmax]

        self.max_layer_index = len(self.all_layers) - 1
        self.neuron_indices = []

    def set_specified_neuron_values(self, index, neuron_indices): # index from 0
        self.max_layer_index = index
        self.neuron_indices = neuron_indices

    def get_neuron_values(self, result):
        if len(self.neuron_indices) > 0:
            result = result[0]
            for idx in self.neuron_indices:
                result = result[idx]
        return result

    def call(self, x):
        for i in range(self.max_layer_index + 1):
            x = self.all_layers[i](x)
        return x

    def get_relevant_layer(self):
        neuron = self.all_layers[self.max_layer_index]
        return neuron






