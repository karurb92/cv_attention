import os

import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import csv
import pandas as pd
from functools import reduce
from time import time
import math

from datasets.siim import SIIM
from data_processing.transforms import *

from appr0_VIT.model import VisionTransformer
from appr0_VIT.solver import Solver
from appr0_VIT.data_generator import DataGenerator
from losses import LDAMLoss, FocalLoss

if __name__ == "__main__":
    # DRW type produced cls weights with values 1. each.
    per_cls_weights = torch.FloatTensor([1.,1.])
    hparams = {
        'batch_size': 32,
        'learning_rate': 1e-3,
        'epochs': 50,
        #'loss_func': torch.nn.BCEWithLogitsLoss(),
        'loss_func':  FocalLoss(weight=per_cls_weights, gamma=2), #more val of gamma means more weight on the misclassified sampls
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
    
    def collate_data(batch):
        batch_dict = {
            'image': torch.stack([img['image'] for img in batch], dim=0).squeeze(1),
            'label': torch.stack([img['label'] for img in batch], dim=0).squeeze()
        }
        return batch_dict

    train_dataloader = DataLoader(dataset=train, batch_size=hparams["batch_size"], shuffle=True,collate_fn=lambda batch: collate_data(batch))
    val_dataloader = DataLoader(dataset=train, batch_size=hparams["batch_size"], shuffle=True,collate_fn=lambda batch: collate_data(batch))
    
    # train_dataloader = DataGenerator(train, batch_size=hparams["batch_size"])
    # val_dataloader = DataGenerator(val, batch_size=hparams["batch_size"])

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

    ########## SAVE MODEL ##########
    os.makedirs('trained_models', exist_ok=True)
    models_path = os.path.join(repo_root, 'trained_models')
    model_name = f'vitmodel_batch{hparams["batch_size"]}_lr{hparams["learning_rate"]}_epochs{hparams["epochs"]}_{math.floor(time())}'
    model.save(os.path.join(models_path, f'{model_name}.model'))

    ########## SAVE STATISTICS ##########
    measurements = {
        'epoch': [i+1 for i in range(len(solver.train_loss_history))],
        'train_loss': solver.train_loss_history,
        'val_loss': solver.val_loss_history,
        'train_TP': solver.train_TP_history,
        'train_FP': solver.train_FP_history,
        'train_TN': solver.train_TN_history,
        'train_FN': solver.train_FN_history,
        'val_TP': solver.val_TP_history,
        'val_FP': solver.val_FP_history,
        'val_TN': solver.val_TN_history,
        'val_FN': solver.val_FN_history,
    }

    measurements = pd.DataFrame(measurements)
    measurements['train_accuracy'] = (measurements['train_TP'] + measurements['train_TN']) / (measurements['train_TP'] + measurements['train_TN'] + measurements['train_FP'] + measurements['train_FN'])
    measurements['train_recall'] = measurements['train_TP'] / (measurements['train_TP'] + measurements['train_FN'])
    measurements['train_precision'] = measurements['train_TP'] / (measurements['train_TP'] + measurements['train_FP'])
    measurements['train_f1'] = 2 * (measurements['train_recall'] * measurements['train_precision']) / (measurements['train_recall'] + measurements['train_precision'])
    measurements['val_accuracy'] = (measurements['val_TP'] + measurements['val_TN']) / (measurements['val_TP'] + measurements['val_TN'] + measurements['val_FP'] + measurements['val_FN'])
    measurements['val_recall'] = measurements['val_TP'] / (measurements['val_TP'] + measurements['val_FN'])
    measurements['val_precision'] = measurements['val_TP'] / (measurements['val_TP'] + measurements['val_FP'])
    measurements['val_f1'] = 2 * (measurements['val_recall'] * measurements['val_precision']) / (measurements['val_recall'] + measurements['val_precision'])

    measurements.to_csv(os.path.join(models_path, f'{model_name}.csv'), sep='|')