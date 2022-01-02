'''
Here we want to define all the transformations that we need to apply to the image.
The idea is to be able to call them in one go with ComposeTransform class (look at the bottom of this file). This will be later used by data generator

- input: 1 image obtained by calling Cifar100 class. (I do not know yet what shape exactly it will have but lets assume numpy 32x32x3)
- output: list of patches transformed (numpy array of dimensionality patches_num x 32 x 32 x 3)

what kind of transforms do we need:
- resizing (some class that would perform resizing of the image). i would say input and output dimensionalities should be both parameters (and not fixed 32x32)
- cutting into patches (Patches class)
- maybe normalization? but let's treat it as second priority, because maybe we can just add 1 special layer at the beginning of the model which is doing it for us

so far I just copied couple of classes that might be a good start
again, this is heavily inspired by I2DL, exercise 1
'''


import numpy as np


class RescaleTransform:
    """Transform class to rescale images to a given range"""
    def __init__(self, out_range=(0, 1), in_range=(0, 255)):
        """
        :param out_range: Value range to which images should be rescaled to
        :param in_range: Old value range of the images
            e.g. (0, 255) for images with raw pixel values
        """
        self.min = out_range[0]
        self.max = out_range[1]
        self._data_min = in_range[0]
        self._data_max = in_range[1]

    def __call__(self, images):
        '''
        Rescales given image
           - from (self._data_min, self._data_max)
           - to (self.min, self.max)
        '''
        images -= self._data_min
        images /= (self._data_max - self._data_min) / (self.max - self.min)
        images += self.min

        return images


class ReshapeToTensor:
    """Transform class to reshape images to a 32x32x3"""
    def __init__(self):
        pass

    def __call__(self, image):
        return image.reshape(3,32,32).transpose(1,2,0)




'''
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


'''