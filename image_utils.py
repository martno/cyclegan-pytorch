
import skimage
import skimage.io
import skimage.color
import numpy as np
import random


def load_image(filepath):
    image = skimage.io.imread(filepath)
    if image.ndim == 2:
        image = skimage.color.gray2rgb(image)

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    assert image.dtype == np.float32, image.dtype
    assert np.min(image) >= 0.0, np.min(image)
    assert np.max(image) <= 1.0, np.max(image)

    return image


def normalize(image):
    return (image - 0.5) / 0.5


def unnormalize(image):
    image = (image * 0.5) + 0.5
    return np.clip(image, 0.0, 1.0)


class ImagePool:
    """
    Saves a list of maximum `size` images. When the pool is full, a randomly selected image might be returned,
    instead of returning the latest stored image.
    """
    def __init__(self, size):
        self.images = []
        self.size = size

    def put(self, image):
        self.images.append(image.detach())

    def get(self):
        assert len(self.images) <= self.size

        if len(self.images) < self.size:
            return self.images[-1]

        if random.random() < 0.5:
            index = -1
        else:
            index = random.randrange(len(self.images))

        return self.images.pop(index)