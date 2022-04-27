import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
import os

class SIIMTrainer(pl.LightningModule):


    def __init__(self, model):
      super().__init__()
      self.model = model


    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
      x, y = train_batch['image'], train_batch['label']
      logits = self.model.forward(x)
      loss = self.cross_entropy_loss(logits, y)
      self.log('train_loss', loss)
      return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch['image'], val_batch['label']
        logits = self.model.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer