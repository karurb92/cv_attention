import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from data_processing.glaucoma import GONRefuge
from data_processing.data_generator import DataGenerator
from data_processing.transforms import *
from approach_1.models import BaselineResNet
from approach_2.glaucoma_training import GlaucomaSolver



if __name__ == "__main__":
    hparams = {
    'batch_size': 3,
    'learning_rate': 1e-3,
    'epochs': 2,
    'loss_func': torch.nn.BCEWithLogitsLoss(),
    'optimizer': optim.AdamW
    }

    num_classes = 2
    freeze=False
    patience = 3
    repo_root = os.path.abspath(os.getcwd())
    data_root = os.path.join(repo_root, "data")
    num_patches = 2 #8

    transforms = [RescaleTransform(), Patches_new(patch_num=num_patches), Resize()]
    # train = Cifar100(root=data_root, purpose='train', seed=seed, split=0.001, transform=transforms)
    # val = Cifar100(root=data_root, purpose='val', seed=seed, split=0.999, transform=transforms)

    train = GONRefuge(root=data_root, purpose='train', transform=transforms)
    val = GONRefuge(root=data_root, purpose='val', transform=transforms)


    train_dataloader = DataGenerator(train, batch_size=hparams['batch_size'])
    # next(iter(train_dataloader))['image'].shpe -> torch.Size([12 (patch x batch_size), 3, 224, 224])

    val_dataloader = DataGenerator(val, batch_size=hparams['batch_size'])
    model = BaselineResNet(num_classes=num_classes, hparams=hparams, freeze=freeze)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    solver = GlaucomaSolver(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        patience=patience,
        approach=1
    )

    solver.train()
