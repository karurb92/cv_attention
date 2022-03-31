#TO BE WRITTEN FROM SCRATCH

#import torch
#from torchvision import transforms
from PIL import Image
import os
import numpy as np
import pandas as pd

class SIIM():
    """Gon Refuge dataset class"""

    def __init__(self, root, purpose, transform=None):
        self.root_path = root
        self.purpose = purpose
        self.images, self.labels = self._make_dataset(directory=self.root_path, purpose=self.purpose)
        self.transform = transform

    @staticmethod
    def _make_dataset(directory, purpose):
        """
        Create the image dataset by preparing a list of samples
        :param directory: root directory of the dataset
        :returns: (images, labels) where:
            - images is a numpy array containing all images in the dataset
            - labels is a list containing one label per image
        """

        if purpose=='train':
            train_1_path = os.path.join(directory, "GON Refuge/Downloaded from  GrandChallenge/REFUGE-Training400/Training400/Glaucoma")
            train_0_path = os.path.join(directory, "GON Refuge/Downloaded from  GrandChallenge/REFUGE-Training400/Training400/Non-Glaucoma")
            images = []
            labels = []
            for file in os.listdir(train_1_path):
                img = Image.open(os.path.join(train_1_path, file))
                #convert_tensor = transforms.ToTensor()
                images.append(np.array(img).astype(float))#convert_tensor(img))
                labels.append(1)
            for file in os.listdir(train_0_path):
                img = Image.open(os.path.join(train_0_path, file))
                #convert_tensor = transforms.ToTensor()
                images.append(np.array(img).astype(float))#convert_tensor(img))
                labels.append(0)
            return images, labels
        elif purpose=='val':
            valid_img_path = os.path.join(directory, "GON Refuge/Downloaded from  GrandChallenge/REFUGE-Validation400/REFUGE-Validation400")
            valid_lab_path = os.path.join(directory, "GON Refuge/Downloaded from  GrandChallenge/REFUGE-Validation400-GT/Fovea_locations.xlsx")
            images = []
            labels = []
            for file in os.listdir(valid_img_path):
                img = Image.open(os.path.join(valid_img_path, file))
                images.append(np.array(img).astype(float))
            labels_excel = pd.read_excel(valid_lab_path)
            labels = labels_excel['Glaucoma Label'].to_list()
            return images, labels
        elif purpose=='test':
            test_img_path = os.path.join(directory, "GON Refuge/Downloaded from  GrandChallenge/REFUGE-Test400")
            test_lab_path = os.path.join(directory, "GON Refuge/Downloaded from  GrandChallenge/REFUGE-Test-GT/Glaucoma_label_and_Fovea_location.xlsx")
            images = []
            labels = []
            for file in os.listdir(test_img_path):
                img = Image.open(os.path.join(test_img_path, file))
                images.append(np.array(img).astype(float))
            labels_excel = pd.read_excel(test_lab_path)
            labels = labels_excel['Label(Glaucoma=1)'].to_list()
            return images, labels


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

        #this needs to have tensors inside
        #The input dimensions are interpreted in the form: mini-batch x channels x [optional depth] x [optional height] x width.
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