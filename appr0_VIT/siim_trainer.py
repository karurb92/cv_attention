import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
import torchmetrics
import os

class SIIMTrainer(pl.LightningModule):


    def __init__(self, model):
      super().__init__()
      self.model = model
      self.save_hyperparameters(model.hparams)
      self.loss_func = model.hparams['loss_func']

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
      x, y = train_batch['image'], train_batch['label']
      logits = self.model.forward(x)
      loss = self.loss_func(logits[:, 1].squeeze(), y.float()) 
      train_acc = torchmetrics.functional.accuracy(logits, y)
      self.log('train_acc', train_acc, on_step=True, on_epoch=False)
      self.log('train_loss', loss,on_step=True)
      return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch['image'], val_batch['label']
        logits = self.model.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        loss = self.loss_func(logits[:, 1].squeeze(), y.float()) 
        val_acc = torchmetrics.functional.accuracy(logits, y)
        self.log('valid_acc', val_acc, on_step=True, on_epoch=True)
        self.log('val_loss', loss,on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)