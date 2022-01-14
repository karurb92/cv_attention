'''
This thing will be generating our batches for the training

- the idea is that it will take subset (train or val) of Cifar100, reshuffle indexes randomly, apply transformations (look at transforms.py) and then will be spitting out batches of images
- our training loop will be using two of such objects (train and validation), which will be then passed to keras' fit function
- i think that for keras fo work with it, either __getitem__ or __iter__ need to be specified

so basically if we use 4 crops then input is for example array N x 32 x 32 x 3 and there should be iterator spitting out batch_size x 4 x 16 x 16 x 3

example of dataloader from I2DL course is below, but we can also look at the one here:  https://github.com/karurb92/ldam_str_bn/blob/main/strat_data_generator.py
these are just for inspiration
'''

import numpy as np
import torch

class DataGenerator:
    """
    Datagenerator Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=True, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        '''
        Defines an iterable function that samples batches from the dataset.
        Each batch is a dict containing numpy arrays of length batch_size (except for the last batch if drop_last=True)
        '''
        def batch_to_torch(batch):
            torch_batch = {}
            for key, value in batch.items():
                if key=='image':
                    torch_batch[key] = torch.tensor(value, dtype=torch.float32)
                elif key=='label':
                    torch_batch[key] = torch.tensor(value, dtype=torch.long)
                else:
                    torch_batch[key] = torch.tensor(value)
            return torch_batch
        
        def combine_batch_dicts(batch):
            batch_dict = {}
            for data_dict in batch:
                for key, value in data_dict.items():
                    if key not in batch_dict:
                        batch_dict[key] = []
                    for val in value:
                        batch_dict[key].append(val)
            return batch_dict
        
        if self.shuffle:
            index_iterator = iter(np.random.permutation(len(self.dataset)))  # define indices as iterator
        else:
            index_iterator = iter(range(len(self.dataset)))  # define indices as iterator

        batch = []
        for i, index in enumerate(index_iterator):  # iterate over indices using the iterator
            #print(i, index)
            batch.append(self.dataset[index])
            #print(batch[0])
            if len(batch) == self.batch_size:
                yield batch_to_torch(combine_batch_dicts(batch))  # use yield keyword to define a iterable generator
                batch = []
            
            if i + 1 == len(self.dataset) and not self.drop_last:
                yield batch_to_torch(combine_batch_dicts(batch))  # use yield keyword to define a iterable generator
                batch = []


    def __len__(self):
        '''
        Returns number of batches obtainable from the dataset
        '''
        return len(self.dataset) // self.batch_size + (1 - int(self.drop_last))



