
from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NN import NN

import tensorflow as tf

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Activation, MaxPooling2D, Dropout


class Alexnet(NN):
  def __init__(self):
    super(Alexnet, self).__init__()
    # OPS
    self.relu = Activation('relu')
    self.maxpool = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid')
    #self.dropout = Dropout(0.4)
    self.softmax = Activation('softmax')

    # Conv layers
    self.conv1 = Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='same')
    self.conv2a = Conv2D(filters=128, kernel_size=(5,5), strides=(1,1), padding='same')
    self.conv2b = Conv2D(filters=128, kernel_size=(5,5), strides=(1,1), padding='same')
    self.conv3 = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same')
    self.conv4a = Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding='same')
    self.conv4b = Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding='same')
    self.conv5a = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same')
    self.conv5b = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same')
    
    # Fully-connected layers

    self.flatten = Flatten()

    self.dense1 = Dense(4096, input_shape=(100,))
    self.dense2 = Dense(4096)
    self.dense3 = Dense(1000)

    # Network definition
  def call(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta = 0.75, bias = 1.0)
    x = self.maxpool(x)

    x = tf.concat((self.conv2a(x[:,:,:,:48]), self.conv2b(x[:,:,:,48:])), 3)
    x = self.relu(x)
    x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta = 0.75, bias = 1.0)
    x = self.maxpool(x)

    x = self.conv3(x)
    x = self.relu(x)
    x = tf.concat((self.conv4a(x[:,:,:,:192]), self.conv4b(x[:,:,:,192:])), 3)
    x = self.relu(x)
    x = tf.concat((self.conv5a(x[:,:,:,:192]), self.conv5b(x[:,:,:,192:])), 3)
    x = self.relu(x)
    x = self.maxpool(x)
    
    x = self.flatten(x)
    
    x = self.dense1(x)
    x = self.relu(x)
    x = self.dense2(x)
    x = self.relu(x)
    x = self.dense3(x)
    
    return self.softmax(x)
