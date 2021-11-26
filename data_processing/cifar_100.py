"""
Definition of Cifar100 dataset class

- the idea is that dataset will be stored as an object which you can call for consecutive images
- if you want to use it, just download original dataset (from toronto site) and unzip it into ".../data/" inside the repository
- if we ever decide to work with another dataset, the only thing that we will need to do is to write such a class for it. the idea is that all the rest stays the same
- if you wanna familiarize yourself more with the concept of having such class, check I2DL course, exercise 3. code below is 100% inspired by it

Karol will finish it
"""

import os
import numpy as np
from PIL import Image
import pickle


class Cifar100():
    """CIFAR-100 dataset class"""
    def __init__(self, root, transform=None):

        self.root_path = root
        self.classes = self._find_classes(directory=self.root_path)
        self.images, self.labels = self._make_dataset(directory=self.root_path)

        # transform function that we will apply later for data preprocessing
        self.transform = transform

    @staticmethod
    def _unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1') #previously 'bytes'
        return dict

    @staticmethod
    def _find_classes(directory):
        """
        Finds all 20 classes in a dataset
        :param directory: root directory of the general data folder
        :returns: dict that maps label to a class
        """
        meta_file = Cifar100._unpickle(f'{directory}\\cifar-100-python\\meta')
        classes = {idx : label.decode('ascii') for idx, label in enumerate(meta_file[b'coarse_label_names'])}
        return classes

    ### TO BE FINISHED
    @staticmethod
    def _make_dataset(directory):
        """
        Create the image dataset by preparing a list of samples
        :param directory: root directory of the dataset
        :returns: (images, labels) where:
            - images is a numpy array containing all images in the dataset
            - labels is a list containing one label per image
        """

        images, labels = [], []
        dirs = ['train', 'test']

        for dir in dirs:
            data_file = Cifar100._unpickle(f'{directory}\\cifar-100-python\\{dir}')
            images.append(data_file[b'data'])
            labels.append(data_file[b'coarse_labels'])
            assert len(images) == len(labels)
            return images, labels

    def __len__(self):
        """Return number of images in the dataset"""
        return(len(self.images))

    ### TO BE FINISHED
    @staticmethod
    def load_image_as_numpy(image_path):
        """Load image from image_path as numpy array"""
        return np.asarray(Image.open(image_path), dtype=float)

    ### TO BE FINISHED
    def __getitem__(self, index):
        data_dict = None
        ########################################################################
        # TODO:                                                                #
        # create a dict of the data at the given index in your dataset         #
        # The dict should be of the following format:                          #
        # {"image": <i-th image>,                                              #
        # "label": <label of i-th image>}                                      #
        # Hints:                                                               #
        #   - use load_image_as_numpy() to load an image from a file path      #
        #   - If applicable (Task 4: 'Transforms and Image Preprocessing'),    #
        #     make sure to apply self.transform to the image:                  #                           
        #     image_transformed = self.transform(image)                        #
        ########################################################################

        data_dict = {}
        if self.transform is None:
            data_dict['image'] = self.load_image_as_numpy(self.images[index])
        else:
            data_dict['image'] = self.transform(self.load_image_as_numpy(self.images[index]))
        data_dict['label'] = self.labels[index]

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return data_dict
