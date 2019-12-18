from Networks.NN import NN
import pkgutil
import inspect
import matplotlib.pyplot as plt
from tensorflow.python.keras.losses import Loss, SparseCategoricalCrossentropy, MSE
from tensorflow.python.keras.backend import argmax
import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os


import zipfile



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~     Weights Loader     ~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
class ImageLoss(Loss):

    def __init__(self):
        super(ImageLoss, self).__init__()



    def call(self, y_true, y_pred):
        l1 = self.cross_entrophy_loss(y_true[:, 0], y_pred[:, 0, :])
        l2 = self.cross_entrophy_loss(y_true[:, 1], y_pred[:, 1, :])
        return l1 + l2



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
    def __init__(self, plots_names, title, output_path):
        self.x = dict()
        self.y = dict()
        self.title = title
        self.output_path = output_path
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
        plt.xlabel("itarations")
        plt.ylabel("value")
        plt.title(self.title)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        plt.savefig(os.path.join(self.output_path, self.title + ".png"))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~   File Managing   ~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def get_module_classes(module, classes, max_depth=2):
    if max_depth > 0:
        for name, obj in inspect.getmembers(module):
            if inspect.ismodule(obj):
                classes.union(classes, get_module_classes(obj, classes, max_depth-1))

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


def preprocess_image(im, crop_size=224):
    im = im.resize([crop_size, crop_size])
    I = np.asarray(im).astype(np.float32)
    I = I[:, :, :3]

    I = np.flip(I, 2)  # BGR
    I = I - [[[104.00698793, 116.66876762, 122.67891434]]]  # subtract mean - whitening
    I = np.reshape(I, (1,) + I.shape).astype(np.float32)
    return I

def create_random_image(size=(224, 224, 3)):
    return Image.fromarray(np.uint8(np.abs(np.random.normal(size=size))*255))

def add_random_noise(image):
    noise = np.random.normal(image.shape())
    result = image + noise
    return result

def unnormalize_image(image):
    image += [[[104.00698793, 116.66876762, 122.67891434]]]
    return np.uint8(np.flip(image, 2))


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = unnormalize_image(np.array(tensor, dtype=np.float32))
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def tensor_to_numpy(tensor):
    return np.array(tensor)

