"""
Definition of Cifar100 dataset class

- the idea is that dataset will be stored as an object which you can call for consecutive images
- if you want to use it, just download original dataset (from toronto site) and unzip it into ".../data/" inside the repository
- if we ever decide to work with another dataset, the only thing that we will need to do is to write such a class for it. the idea is that all the rest stays the same
- if you wanna familiarize yourself more with the concept of having such class, check I2DL course, exercise 3. code below is 100% inspired by it
"""

import numpy as np
import pickle
import random


class Cifar100():
    """CIFAR-100 dataset class"""

    def __init__(self, root, purpose, seed, split=0.8, transform=None):
        self.root_path = root
        self.seed = seed
        self.split = split
        self.purpose = purpose
        self.classes = self._find_classes(directory=self.root_path)
        self.images, self.labels = self._make_dataset(directory=self.root_path, purpose=self.purpose, split=self.split, seed=self.seed)
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
        meta_file = Cifar100._unpickle(f'{directory}/cifar-100-python/meta')
        classes = {idx : label for idx, label in enumerate(meta_file['coarse_label_names'])}
        return classes

    @staticmethod
    def _make_dataset(directory, purpose, split, seed):
        """
        Create the image dataset by preparing a list of samples
        :param directory: root directory of the dataset
        :returns: (images, labels) where:
            - images is a numpy array containing all images in the dataset
            - labels is a list containing one label per image
        """

        if purpose=='train':
            data_file = Cifar100._unpickle(f'{directory}/cifar-100-python/train')
            n = len(data_file['data'])
            random.seed(seed)
            random_mask = random.sample([i for i in range(n)], int(split*n))
            images = np.array(data_file['data'])[random_mask].astype(float)
            labels = np.array(data_file['coarse_labels'])[random_mask]
            return images, labels
        elif purpose=='val':
            data_file = Cifar100._unpickle(f'{directory}/cifar-100-python/train')
            n = len(data_file['data'])
            random.seed(seed)
            random_mask = random.sample([i for i in range(n)], int(split*n))
            images = np.delete(np.array(data_file['data']), random_mask, axis=0).astype(float)
            labels = np.delete(np.array(data_file['coarse_labels']), random_mask, axis=0)
            return images, labels
        elif purpose=='test':
            data_file = Cifar100._unpickle(f'{directory}/cifar-100-python/test')
            return np.array(data_file['data']).astype(float), data_file['coarse_labels']


    def __len__(self):
        """Return number of images in the dataset"""
        return(len(self.images))


    def __getitem__(self, index):
        """
        Creates a dict of the data at the given index:
            {"image": <i-th image>,                                              #
             "label": <label of i-th image>} 
        """

        data_dict = {}

        data_dict['image'] = self.images[index]
        data_dict['label'] = self.labels[index]

        for transform in self.transform:
            data_dict = transform(data_dict)
        
        '''
        if self.transform is None:
            data_dict['image'] = self.images[index]
        else:
            for transform in self.transform:
                data_dict['image'] = transform(self.images[index])
        
        data_dict['label'] = self.labels[index]
        '''

        return data_dict
