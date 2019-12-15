from Networks.NN import NN
import pkgutil
import inspect
import matplotlib.pyplot as plt
from tensorflow.python.keras.losses import Loss, SparseCategoricalCrossentropy, MSE
from tensorflow.python.keras.backend import argmax
import tensorflow as tf
import numpy as np
import sys


import zipfile



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~     Weights Loader     ~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
class AlexNetWeightsLoader:
    def load(self, model, wdir):
        model.conv1.set_weights((np.load(wdir + 'conv1.npy'), np.load(wdir + 'conv1b.npy')))
        model.conv2a.set_weights((np.load(wdir + 'conv2_a.npy'), np.load(wdir + 'conv2b_a.npy')))
        model.conv2b.set_weights((np.load(wdir + 'conv2_b.npy'), np.load(wdir + 'conv2b_b.npy')))
        model.conv3.set_weights((np.load(wdir + 'conv3.npy'), np.load(wdir + 'conv3b.npy')))
        model.conv4a.set_weights((np.load(wdir + 'conv4_a.npy'), np.load(wdir + 'conv4b_a.npy')))
        model.conv5a.set_weights((np.load(wdir + 'conv5_a.npy'), np.load(wdir + 'conv5b_a.npy')))
        model.conv4b.set_weights((np.load(wdir + 'conv4_b.npy'), np.load(wdir + 'conv4b_b.npy')))
        model.conv5b.set_weights((np.load(wdir + 'conv5_b.npy'), np.load(wdir + 'conv5b_b.npy')))

        model.dense1.set_weights((np.load(wdir + 'dense1.npy'), np.load(wdir + 'dense1b.npy')))
        model.dense2.set_weights((np.load(wdir + 'dense2.npy'), np.load(wdir + 'dense2b.npy')))
        model.dense3.set_weights((np.load(wdir + 'dense3.npy'), np.load(wdir + 'dense3b.npy')))
        return model



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~     Plotting      ~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

class Plotter:
    def __init__(self, plots_names, network_name, title):
        self.network_name = network_name
        self.x = dict()
        self.y = dict()
        self.title = title
        for name in plots_names:
            self.x[name] = []
            self.y[name] = []

    def add(self, name, x, y):
        self.x[name].append(x)
        self.y[name].append(y)

    def plot(self):
        plt.figure()
        for name in self.x.keys():
            plt.plot(self.x[name], self.y[name], label=name)
        plt.legend()
        plt.title(self.title + " for " + self.network_name + " network")
        plt.savefig(self.network_name + " " + self.title + ".png")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~   File Managing   ~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def get_module_classes(module, classes):
    for name, obj in inspect.getmembers(module):

        if inspect.ismodule(obj):
            classes.union(classes, get_module_classes(obj, classes))

        if inspect.isclass(obj) and issubclass(obj, NN):
            classes.add(obj)
    return classes


def get_object(object_type, package, *args):
    classes = set()
    prefix = package.__name__ + "."
    for importer, modname, ispkg in pkgutil.iter_modules(package.__path__, prefix):
        # print("Found submodule %s (is a package: %s)" % (modname, ispkg))
        module = __import__(modname)
        result = get_module_classes(module, set())
        classes = classes.union(result)

    for class_obj in classes:
        if object_type == class_obj.__name__:
            return class_obj(*args)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~   Datasets loaders   ~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def preprocess_image(im):
    im = im.resize([224, 224])
    I = np.asarray(im).astype(np.float32)
    I = I[:, :, :3]

    I = np.flip(I, 2)  # BGR
    I = I - [[[104.00698793, 116.66876762, 122.67891434]]]  # subtract mean - whitening
    I = np.reshape(I, (1,) + I.shape)
    return I


def extract_zip(path_to_zip_file, directory_to_extract_to):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
