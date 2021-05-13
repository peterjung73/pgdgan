# Files of this project is modified versions of 'https://github.com/AshishBora/csgm', which
#comes with the MIT licence: https://github.com/AshishBora/csgm/blob/master/LICENSE

import dcgan_utils
import numpy as np
import utils
import scipy.misc


def display_transform(image):
    image = dcgan_utils.inverse_transform(image)
    return image


def view_image(image, hparams, mask=None):
    """Process and show the image"""
    image = display_transform(image)
    if len(image) == hparams.n_input:
        image = image.reshape(hparams.image_shape)
        if mask is not None:
            mask = mask.reshape(hparams.image_shape)
            image = np.maximum(np.minimum(1.0, image - 1.0*image*(1-mask)), -1.0)
    utils.plot_image(image)


def save_image(image, path):
    """Save an image as a png file"""
    image = dcgan_utils.inverse_transform(image)
    with open(path, 'wb') as outfile:
        scipy.misc.imsave(outfile, 255*image)