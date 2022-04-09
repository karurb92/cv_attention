import torch
from PIL import Image
from torchvision import transforms
#import torchvision
import os
import numpy as np
import pandas as pd

class SIIM():
    """Gon Refuge dataset class"""

    def __init__(self, root, purpose, seed, split, transform=None):
        self.root_path = root
        self.purpose = purpose
        self.seed = seed
        self.split = split
        self.images, self.labels = self._make_dataset(directory=self.root_path, purpose=self.purpose, seed=self.seed, split=self.split)
        self.transform = transform

    @staticmethod
    def _make_dataset(directory, purpose, seed, split):
        """
        Create the image dataset by preparing a list of samples
        :param directory: root directory of the dataset
        :returns: (images, labels) where:
            - images is a numpy array containing all images in the dataset
            - labels is a list containing one label per image
        """

        data_path = os.path.join(directory, "train.csv")
        meta_df = pd.read_csv(data_path, sep=',')

        #do we want to apply stratification here?
        train, val, test = np.split(meta_df.sample(frac=1, random_state=seed), 
                                        [int(split*meta_df.shape[0]), int(((1.0-split)/2.0+split)*meta_df.shape[0])])

        ############### this here needs to go
        '''
        if purpose=='train':
            return ['ISIC_0015719','ISIC_0052212','ISIC_0075663','ISIC_0076545'], [1,0,0,0]
        elif purpose=='val':
            return ['ISIC_0079038', 'ISIC_0084270'], [1,0]
        '''
        ######################

        if purpose=='train':
            return train['image_name'].tolist(), train['target'].tolist()
        elif purpose=='val':
            return val['image_name'].tolist(), val['target'].tolist()
        elif purpose=='test':
            return test['image_name'].tolist(), test['target'].tolist()


    def __len__(self):
        """Return number of images in the dataset"""
        return(len(self.images))


    def __getitem__(self, index):
        """
        Creates a dict of the data at the given index:
            {"image": <i-th image>,                                              #
             "label": <label of i-th image>} 
        """

        img_root = os.path.join(self.root_path, f"jpeg/train/{self.images[index]}.jpg")
        img = Image.open(img_root)
        trans = transforms.ToTensor()
        img = trans(img)
        #img = torchvision.io.read_image(img_root)

        data_dict = {
            'image': img.unsqueeze(0),
            'label': torch.tensor([self.labels[index]])
        }

        if self.transform is None: return data_dict

        for transform in self.transform:
            data_dict = transform(data_dict)

        return data_dict