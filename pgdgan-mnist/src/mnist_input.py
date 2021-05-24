# Files of this project is modified versions of 'https://github.com/AshishBora/csgm', which
#comes with the MIT licence: https://github.com/AshishBora/csgm/blob/master/LICENSE

import glob
import numpy as np
import gan_utils
import mnist_model_def
import tensorflow as tf


def get_full_input(hparams):
    """Create input tensors"""
    image_paths = glob.glob(hparams.input_path_pattern)
    if hparams.input_type == 'full-input':
        image_paths.sort()
        image_paths = image_paths[:hparams.num_input_images]
    elif hparams.input_type == 'random-test':
        idxs = np.random.choice(len(image_paths), hparams.num_input_images)
        image_paths = [image_paths[idx] for idx in idxs]
    else:
        raise NotImplementedError

    #def get_image(image_path, image_size, is_crop=True, resize_w=28, is_grayscale=True):
    #im = gan_utils.get_image('./data/mnist-test/039.png',20)
    #h,w = im.shape
    images = [gan_utils.get_image(image_path,64) for image_path in image_paths]
    images = {i: image.reshape([28*28]) for (i, image) in enumerate(images)}
    return images


def sample_generator_images(hparams):
    """Sample random images from the generator"""

    # Get a session
    sess = tf.Session()

    # Create the generator
    z_batch = tf.Variable(tf.random_normal([hparams.batch_size, 20]))
    x_hat_batch, restore_dict, restore_path = mnist_model_def.gan_gen(z_batch, hparams)

    # Intialize and restore model parameters
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    restorer = tf.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)
    images = sess.run(x_hat_batch)
    images = {i: image for (i, image) in enumerate(images)}

    # Reset TensorFlow graph
    sess.close()
    tf.reset_default_graph()

    return images


def model_input(hparams):
    """Create input tensors"""

    if hparams.input_type == 'full-input':
        images = get_full_input(hparams)
    elif hparams.input_type == 'random-test':
        images = get_full_input(hparams)
    elif hparams.input_type == 'gen-span':
        images = sample_generator_images(hparams)
    else:
        raise NotImplementedError

    return images
