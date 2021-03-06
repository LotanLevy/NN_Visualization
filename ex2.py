from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import tensorflow as tf
import numpy as np
from PIL import ImageFilter
from classes import classes
import seaborn as sns; sns.set()

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
    parser.add_argument('--question', '-q', type=int, default=1, help='question number')

    parser.add_argument('--nntype', default="Alexnet", help='The type of the network')
    parser.add_argument('--orig_image_path', help='The path of image to read')
    parser.add_argument('--image_path', default=None, help='The path to keep the learned image')

    parser.add_argument('--crop_size', type=int, default=224, help='The crop size of the image')
    parser.add_argument('--ckpt_path', default="weights", help='The type of the network')

    # q1-q3
    parser.add_argument('--optimizer', '-opt', default="adam", help='optimizer  type')
    parser.add_argument('--max_iter', type=int, default=100, help='The maximum iterations')
    parser.add_argument('--print_freq', '-pf', type=int, default=500, help='The printing frequency')
    parser.add_argument('--max_pred_value', '-pv', type=float, default=0.95, help='Max prediction value')
    parser.add_argument('--reg_factor', type=float, default=0.2, help='The regression (lambda) value')
    parser.add_argument('--reg_type', default="Basic", help='The regression type (Fourier/basic)')
    parser.add_argument('--neuron_layer_idx', "-nl", type=int, default=21, help='The index of the require neuron')
    parser.add_argument('-ni', '--neuron_idx_list', type=int, default=[], action='append', help='The indices of the neuron (-ni=1, -ni=2 -ni=3)')

    # q4
    parser.add_argument('--stride', '-s', type=int, default=2, help='The stride of the zeros block')
    parser.add_argument('--kernel_size', '-ks', type=int, default=2, help='The stride of the zeros block')


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
        return tf.keras.optimizers.Adam(0.15)
    return None


def train_main(max_iterations, image_trainer, trained_image, print_freq, max_pred_val, plot_title, plot_path):
    trained_image = tf.Variable(trained_image)
    loss_plotter = Plotter(["loss"], plot_title + "[loss]", plot_path)
    pred_plotter = Plotter(["prediction"], plot_title + "[prediction]", plot_path)

    train_step = image_trainer.get_step()
    i = 0
    for _ in range(max_iterations):
        i += 1
        train_step(trained_image)
        if image_trainer.last_pred.result().numpy() > max_pred_val:
            print("trainer achieved the maximum value")
            break
        if i%print_freq == 0:
            print("loss after {} iterations: {}, prediction {}".format(i + 1,
                                  image_trainer.train_loss.result(), image_trainer.last_pred.result()))
            loss_plotter.add("loss", i+1, image_trainer.train_loss.result())
            pred_plotter.add("prediction", i+1, image_trainer.last_pred.result())
    print("Training is stop after {} iterations".format(i))

    loss_plotter.plot()
    pred_plotter.plot()
    return trained_image


def basic_visualization(args):
    args.reg_type = "Basic"
    visualization_by_args(args)


def fourier_visualization(args):

    args.reg_type = "Fourier"
    visualization_by_args(args)


def visualization_by_args(args):
    print("running: ", args)
    tf.keras.backend.set_floatx('float32')

    # optimizer
    optimizer = get_optimizer(args.optimizer)

    # Loading the image and pre-processing it
    if args.orig_image_path is None:
        im = create_random_image()
    else:
        im = Image.open(args.orig_image_path)
    I = preprocess_image(im, args.crop_size)


    # Build the network and loads its weights
    model = get_network(args.nntype)
    model.set_specified_neuron_values(args.neuron_layer_idx, args.neuron_idx_list)  # specify The neuron to visualize
    model(I)# Init graph
    weights_loader = AlexNetWeightsLoader()
    weights_loader.load(model, args.ckpt_path + "/")

    neuron_repre = ' '.join(map(str, args.neuron_idx_list))

    result_title = "result_with_reg_type_{}_for_layer_num_{}_neuron_{}".format(args.reg_type, args.neuron_layer_idx, neuron_repre)

    # Build an image trainer object
    trainer = ImageTrainer(model, optimizer, args.reg_type, args.reg_factor)

    output_path = os.path.join(args.image_path, "learned_images")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # The Training process
    learned_result = train_main(args.max_iter, trainer,np.copy(I), args.print_freq, args.max_pred_value, result_title, output_path)

    # convert network output into image and save the results
    scale_factor = 255 if args.neuron_layer_idx == (len(model.all_layers) - 1) else 1
    learned_image = tensor_to_image(learned_result, scale_factor)

    im.save(os.path.join(output_path, "reg_type_{}_orig_for_layer_num_{}_neuron_{}.png".format(args.reg_type, args.neuron_layer_idx, neuron_repre)))
    learned_image.save(os.path.join(output_path, "{}.png".format(result_title)))


    pred_model = get_network(args.nntype)
    pred_model(I)# Init graph
    weights_loader.load(pred_model, args.ckpt_path + "/")

    result_c = tf.cast(pred_model(learned_result), tf.float32).numpy()
    orig_c = tf.cast(pred_model(I), tf.float32).numpy()

    result_ind = np.argmax(result_c)
    orig_ind = np.argmax(orig_c)

    print("result: %d, %s, score %f" % (result_ind, classes[result_ind], np.max(result_c)))
    print("orig: %d, %s, score %f" % (orig_ind, classes[orig_ind], np.max(orig_c)))


    print("End process")


def cls_heat_map(args):
    im = Image.open(args.orig_image_path)
    im.show()
    I = preprocess_image(im, args.crop_size)
    model = get_network(args.nntype)
    model(I)  # Init graph
    weights_loader = AlexNetWeightsLoader()
    weights_loader.load(model, args.ckpt_path + "/")

    output_path = os.path.join(args.image_path, "learned_images")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    result = model(I)

    pred = tf.cast(result, tf.float32).numpy()
    best_cls = np.argmax(pred)

    dim = int((I.shape[1] - args.kernel_size)/args.stride)

    h_m = np.zeros((dim, dim))
    i,j=0,0
    r,c = int(np.floor(args.kernel_size/2)), int(np.floor(args.kernel_size/2))
    while r < I.shape[1] - int(np.ceil(args.kernel_size/2)) :
        while c < I.shape[2] - int(np.ceil(args.kernel_size/2)):
            image = I.copy()
            image[0, r - int(np.floor(args.kernel_size/2)): r + int(np.ceil(args.kernel_size/2)), c - int(np.floor(args.kernel_size/2)): c + int(np.ceil(args.kernel_size/2)), :] = 0

            cls_score = tf.cast(model(image), tf.float32).numpy()[0, best_cls]
            h_m[i,j] = cls_score
            c += args.stride
            j+=1
        r += args.stride
        i+=1
        j=0
        c = int(np.floor(args.kernel_size/2))

    print(h_m)

    plt.figure()
    ax = sns.heatmap(h_m, vmin=0, vmax=1)
    title = "Q4_class_{}_with_kernel_{}_and_stride_{}".format(classes[best_cls], args.kernel_size, args.stride)
    plt.title(title)
    plt.savefig(os.path.join(output_path, title + ".png"))
    plt.show()












if __name__ == '__main__':
    args = get_args()
    if args.question < 4:
        visualization_by_args(args)
    else:
        cls_heat_map(args)


