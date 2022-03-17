import os

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim


from data_processing.cifar_100 import Cifar100
from data_processing.data_generator import DataGenerator
from data_processing.transforms import *
from approach_1.models import VisionTransformer
from approach_1.attention_training import AttentionSolver


if __name__ == "__main__":
    hparams = {
        'batch_size': 8,
        'learning_rate': 1e-3,
        'epochs': 2,
        'loss_func': torch.nn.CrossEntropyLoss(),
        'optimizer': optim.AdamW
    }

    repo_root = os.path.abspath(os.getcwd())
    model_root = os.path.join(repo_root, "trained_models/baselinemodel_batch1_lr0.001_epochs2_freezeFalse.model")
    data_root = os.path.join(repo_root, "data")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 69
    split = 0.7
    patch_size = 16
    num_patches = int((32/patch_size)**2)
    num_classes = 20

    cnn_model = torch.load(model_root)
    cnn_model = nn.Sequential(*(nn.ModuleList(cnn_model.children())[:-2]).to(device))

    transforms = [RescaleTransform(), ReshapeToTensor(), Patches(patch_size=patch_size), Resize(), PassThroughCNN(cnn_model)]
    train = Cifar100(root=data_root, purpose='train', seed=seed, split=0.01, transform=transforms)
    val = Cifar100(root=data_root, purpose='val', seed=seed, split=0.999, transform=transforms)

    train_dataloader = DataGenerator(train, batch_size=hparams["batch_size"], flatten_batch=False)
    val_dataloader = DataGenerator(val, batch_size=hparams["batch_size"], flatten_batch=False)

    flattened_dim = train[0]['image'].flatten(2,3).flatten(1,2).shape[1]

    model = VisionTransformer(**{
                                    'embed_dim': 256,
                                    'hidden_dim': 512,
                                    #'num_heads': 8,
                                    #'num_layers': 6,
                                    'num_heads': 4,
                                    'num_layers': 2,
                                    'patch_size': patch_size,
                                    'flattened_dim': flattened_dim,
                                    'num_patches': num_patches,
                                    'num_classes': num_classes,
                                    'dropout': 0.2,
                                    'hparams': hparams
                                })

    patience = 3


    solver = AttentionSolver(
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
