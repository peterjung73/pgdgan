# pylint: disable = C0103, C0111, C0301, R0914

"""Model definitions for celebA

This file is partially based on
https://github.com/carpedm20/DCGAN-tensorflow/blob/master/main.py
https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py

They come with the following license: https://github.com/carpedm20/DCGAN-tensorflow/blob/master/LICENSE
"""

import tensorflow as tf
#import ops
import mnist_gan.ops as ops

class Hparams(object):
    def __init__(self):
        self.c_dim = 3
        self.z_dim = 100
        self.gf_dim = 64
        self.df_dim = 64
        self.gfc_dim = 1024
        self.dfc_dim = 1024
        self.batch_size = 64


# xavier initialization
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def generator(z):

    # initialize weights and biases for generator
    G_W1 = tf.Variable(xavier_init([20, 200]), name="G_W1")
    G_b1 = tf.Variable(tf.zeros(shape=[200]), name="G_b1")

    G_W2 = tf.Variable(xavier_init([200, 784]), name="G_W2")
    G_b2 = tf.Variable(tf.zeros(shape=[784]), name="G_b2")

    theta_G = [G_W1, G_W2, G_b1, G_b2]

    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_W1 = tf.Variable(xavier_init([784, 128]), name="D_W1")
    D_b1 = tf.Variable(tf.zeros(shape=[128]), name="D_b1")

    D_W2 = tf.Variable(xavier_init([128, 1]), name="D_W2")
    D_b2 = tf.Variable(tf.zeros(shape=[1]), name="D_b2")

    theta_D = [D_W1, D_W2, D_b1, D_b2]

    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit




def gen_restore_vars():

    restore_vars = [
                    'Z',
                    'G_W1',
                    'G_W2',
                    'G_b1',
                    'G_b2']

    return restore_vars


def discrim_restore_vars():
    restore_vars = [
        "X",
        "D_W1",
        "D_W2",
        "D_b1",
        "G_b2"]
    return restore_vars
