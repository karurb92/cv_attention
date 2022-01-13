'''
There should be 2 classes here:

- one for CNN model (something inheriting from already pretrained resnet or sth, WITH the last layer included). we will want to train it on our crops
- one for attention model (this i guess should take CNN model as input, cut off the last layer, calculate the embeddings and then feed it to actual multihead attention layers). This part actually might be quite challenging

below is an inspiration (in form of function instead of class)
'''

import torch
import torch.nn as nn
import torchvision.models as models

class BaselineResNet(nn.Module):

    def __init__(self, num_classes=20, hparams=None):
        super().__init__()

        self.hparams=hparams
        self.num_classes = num_classes
        
        self.feature_extractor = nn.Sequential(*(list(models.resnet18(pretrained=True).children())[:-2]))
        self.AdAvgP = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.FC = nn.Sequential(nn.Linear(in_features=512, out_features=self.num_classes, bias=True))

        '''
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        ''' 

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        
        x = self.feature_extractor(x)
        x = self.AdAvgP(x)
        x = x.squeeze()
        x = self.FC(x)
        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)