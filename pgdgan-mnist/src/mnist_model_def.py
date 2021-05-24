# Files of this project is modified versions of 'https://github.com/AshishBora/csgm', which
#comes with the MIT licence: https://github.com/AshishBora/csgm/blob/master/LICENSE

"""Model definitions for mnist

This file is partially based on
https://github.com/carpedm20/gan-tensorflow/blob/master/main.py
https://github.com/carpedm20/gan-tensorflow/blob/master/model.py

They come with the following license: https://github.com/carpedm20/gan-tensorflow/blob/master/LICENSE
"""

import os
import sys
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mnist_gan import model_def as mnist_gan_model_def


def gan_discrim(x_hat_batch, hparams):

    assert hparams.batch_size in [1, 4], 'batch size should be either 4 or 1'
    #x_hat_image = tf.reshape(x_hat_batch, [-1, 28, 28])
    x_hat_image = x_hat_batch
    all_zeros = tf.zeros([4, 784])
    discrim_input = all_zeros + x_hat_image

    prob, _ = mnist_gan_model_def.discriminator(discrim_input)
    prob = tf.reshape(prob, [-1])
    prob = prob[:hparams.batch_size]

    restore_vars = mnist_gan_model_def.gen_restore_vars()
    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
    restore_path = tf.train.latest_checkpoint(hparams.pretrained_model_dir)

    return prob, restore_dict, restore_path


def gan_gen(z, hparams):

    assert hparams.batch_size in [1, 4], 'batch size should be either 4 or 1'
    z_full = tf.zeros([4, 20]) + z

    x_hat_full = mnist_gan_model_def.generator(z_full)
    x_hat_batch = tf.reshape(x_hat_full[:hparams.batch_size], [hparams.batch_size, 28*28*1])

    restore_vars = mnist_gan_model_def.gen_restore_vars()
    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
    restore_path = tf.train.latest_checkpoint(hparams.pretrained_model_dir)

    return x_hat_batch, restore_dict, restore_path
