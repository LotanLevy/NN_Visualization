from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import tensorflow as tf
import numpy as np
from PIL import Image
from classes import classes

from utils import *
from train_test import ImageTrainer
import os




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
    parser.add_argument('--optimizer', '-opt', default="adam", help='optimizer  type')

    # parser.add_argument('--dstype', default="num", help='The type of the dataset')
    parser.add_argument('--ckpt_path', default="weights", help='The type of the network')
    parser.add_argument('--image_path', default="", help='The path to keep the learned image')
    parser.add_argument('--reg_factor', type=float, default=0.00002, help='The regression (lambda) value')
    parser.add_argument('--min_target_activation', type=float, default=400, help='The min value for the neuron activation')
    parser.add_argument('--max_iter', type=int, default=100, help='The maximum iterations')
    parser.add_argument('--neuron_layer_idx', "-nl", type=int, default=21, help='The index of the require neuron')
    parser.add_argument('-ni', '--neuron_idx_list', type=int, default=[0], action='append', help='The indices of the neuron (-ni=1, -ni=2 -ni=3)')






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

def get_optimizer(optimizer_type):
    """
    :param optimizer_type: The require optimizer type
    :return: optimizer object
    """
    if optimizer_type == "adam":
        return tf.keras.optimizers.Adam()
    return None




def train_main(max_iterations, image_trainer, trained_image):
    trained_image = tf.Variable(trained_image)
    train_step = image_trainer.get_step()
    iter_counter = 0
    for i in range(max_iterations):
        iter_counter += 1
        train_step(trained_image)
        if image_trainer.end_training:
            print("trainer achieved the maximum value")
            break
        if i%10==0:
            print("loss after {} iterations: {}, prediction {}".format(i + 1,
                                  image_trainer.train_loss.result(), image_trainer.last_pred.result()))
    print("Training is stop after {} iterations".format(iter_counter))
    return trained_image



def main_by_args(args):
    tf.keras.backend.set_floatx('float32')
    weights_loader = AlexNetWeightsLoader()
    model = get_network(args.nntype)
    optimizer = get_optimizer(args.optimizer)
    # im = Image.open("images/poodle.png")
    im = create_random_image()
    im.show()
    I = preprocess_image(im)

    model(I)  # Init graph
    weights_loader.load(model, args.ckpt_path + "/")
    model.set_specified_neuron_values(args.neuron_layer_idx, args.neuron_idx_list)
    trainer = ImageTrainer(model, optimizer, args.min_target_activation, args.reg_factor)

    learned_image = train_main(args.max_iter, trainer, I)

    learned_image = tensor_to_image(learned_image)



    output_path = "learned_images"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    im.save("learned_images/orig_for_layer_num_{}_neuron_{}.png".format(args.neuron_layer_idx, ' '.join(map(str, args.neuron_idx_list))))
    learned_image.save("learned_images/image_for_layer_num_{}_neuron_{}.png".format(args.neuron_layer_idx, ' '.join(map(str, args.neuron_idx_list)) ))


    # c = model(I)
    #
    # top_ind = np.argmax(c)
    # print("Top1: %d, %s" % (top_ind, classes[top_ind]))




if __name__ == '__main__':
    args = get_args()
    main_by_args(args)