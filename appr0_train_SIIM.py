import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from functools import reduce

from datasets.siim import SIIM
from data_processing.transforms import *

from appr0_VIT.model import VisionTransformer
from appr0_VIT.solver import Solver
from appr0_VIT.data_generator import DataGenerator


if __name__ == "__main__":
    hparams = {
        'batch_size': 32,
        'learning_rate': 1e-3,
        'epochs': 50,
        'loss_func': torch.nn.BCEWithLogitsLoss(),
        'optimizer': optim.AdamW,
        'patch_num': 8,
        'new_size': (3, 400, 500)
    }

    repo_root = os.path.abspath(os.getcwd())
    model_root = os.path.join(repo_root, "trained_models")
    data_root = os.path.join(repo_root, "data/siim")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 42
    split = 0.7
    num_classes = 2

    transforms = [Patches(patch_num=hparams['patch_num']), Resize(new_size=hparams['new_size'])]

    train = SIIM(root=data_root, purpose='train', seed=seed, split=split, transform=transforms)
    val = SIIM(root=data_root, purpose='val', seed=seed, split=split, transform=transforms)

    train_dataloader = DataGenerator(train, batch_size=hparams["batch_size"])
    val_dataloader = DataGenerator(val, batch_size=hparams["batch_size"])

    flattened_dim = reduce((lambda x, y: x * y), hparams['new_size'])

    model = VisionTransformer(**{
                                    'embed_dim': 256,
                                    'hidden_dim': 512,
                                    #'num_heads': 8,
                                    #'num_layers': 6,
                                    'num_heads': 4,
                                    'num_layers': 2,
                                    'flattened_dim': flattened_dim,
                                    'num_patches': hparams['patch_num']**2,
                                    'num_classes': num_classes,
                                    'dropout': 0.2,
                                    'hparams': hparams
                                })

    patience = 20

    solver = Solver(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        patience=patience
    )

    solver.train()

    os.makedirs('trained_models', exist_ok=True)
    models_path = os.path.join(repo_root, 'trained_models')
    model.save(os.path.join(models_path, f'vitmodel_batch{hparams["batch_size"]}_lr{hparams["learning_rate"]}_epochs{hparams["epochs"]}.model'))
