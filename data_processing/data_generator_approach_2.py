import numpy as np
import torch

class DataGeneratorA2:
    """
    Datagenerator Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=True, drop_last=False, flatten_batch=True):
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
        self.flatten_batch = flatten_batch

    def __iter__(self):
        '''
        Defines an iterable function that samples batches from the dataset.
        Each batch is a dict containing numpy arrays of length batch_size (except for the last batch if drop_last=True)
        '''
        def batch_to_torch(batch):
            torch_batch = {}
            for key, value in batch.items():
                if key=='image':
                    if self.flatten_batch:
                        torch_batch[key] = torch.tensor(value, dtype=torch.float32)
                    else:
                        torch_batch[key] = torch.cat([x.float().unsqueeze(0) for x in value_tensor], dim=0).flatten(3,4).flatten(2,3)
                elif key=='label':
                    torch_batch[key] = torch.tensor(value, dtype=torch.long)
                else:
                    torch_batch[key] = torch.tensor(value)
            return torch_batch
        
        def combine_batch_dicts(batch):
            #(4,512,7,7)
            batch_dict = {}
            for data_dict in batch:
                for key, value in data_dict.items():
                    if key not in batch_dict:
                        batch_dict[key] = []
                    if self.flatten_batch:
                        #for val in value:
                        batch_dict[key].append(value)
                    else:
                        if key=='image':
                            batch_dict[key].append(value)
                        elif key=='label':
                            batch_dict[key].append(value[0]) 
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