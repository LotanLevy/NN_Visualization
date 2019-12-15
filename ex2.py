from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import tensorflow as tf
import numpy as np
from PIL import Image
from classes import classes

from utils import get_object, Plotter, AlexNetWeightsLoader, preprocess_image
from train_test import Trainer, Validator


"""
This file manage all the exercise questions. 
Each question can be run with different configurations in the execution command. 
1.2 - python ex2.py --nntype=Basic --dstype=num --plot_freq=500 --epochs=14
2 - python ex2.py --nntype=Linear --dstype=num --plot_freq=500 --epochs=14
3.1 - python ex2.py --nntype=Shallow --dstype=num --plot_freq=500 --epochs=14
3.2 python ex2.py --nntype=ReducedBasic --dstype=num --plot_freq=500 --epochs=14
4.1 python ex2.py --nntype=Basic --dstype=num --ts=250 --plot_freq=10 --epochs=40
4.2 python ex2.py --nntype=Overfit --dstype=num --ts=250 --plot_freq=10 --epochs=40
4.1 python ex2.py --nntype=DropoutBasic --dstype=num --ts=250 --plot_freq=10 --epochs=40
5.1 python ex2.py --nntype=SumNet --dstype=sum --plot_freq=500 --epochs=14
5.1 python ex2.py --nntype=SumNet --dstype=sum --plot_freq=500 --epochs=14
5.2 python ex2.py --nntype=PairNet --dstype=pair --plot_freq=500 --epochs=14 --loss=pairloss
"""


def get_args():
    """
    Reads the execution configurations.
    :return: The configuration values in an argparse object
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--nntype', default="Alexnet", help='The type of the network')
    # parser.add_argument('--dstype', default="num", help='The type of the dataset')
    parser.add_argument('--ckpt_path', default="weights", help='The type of the network')
    return parser.parse_args()


def get_network(network_type):
    """
    :param network_type: The require network type
    :return: A network object according to the given network type.
    The program creates only objects that inherits from the class NN.
    """
    import Networks
    package = Networks
    return get_object(network_type, package)




def main_by_args(args):
    tf.keras.backend.set_floatx('float64')

    weights_loader = AlexNetWeightsLoader()

    model = get_network(args.nntype)

    im = Image.open("poodle.png")
    I = preprocess_image(im)

    model(I)  # Init graph
    weights_loader.load(model, args.ckpt_path + "/")

    c = model(I)

    top_ind = np.argmax(c)
    print("Top1: %d, %s" % (top_ind, classes[top_ind]))




if __name__ == '__main__':
    args = get_args()
    main_by_args(args)



