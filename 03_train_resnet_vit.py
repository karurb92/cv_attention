import os

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from data_processing.glaucoma import GONRefuge
from data_processing.data_generator_approach_2 import DataGeneratorA2
from data_processing.transforms import *
from approach_2.model import VisionTransformerEmbedded
from approach_2.glaucoma_training import GlaucomaSolver

if __name__ == "__main__":
    hparams = {
        'batch_size': 5,
        'learning_rate': 1e-5,
        'epochs': 50,
        'loss_func': torch.nn.BCEWithLogitsLoss(),
        'optimizer': optim.AdamW
    }

    repo_root = os.path.abspath(os.getcwd())
    data_root = os.path.join(repo_root, "data")

    patch_size = 16 #this is not used either way
    num_patches = 8
    num_classes = 2

    transforms = [RescaleTransform(), Patches_new(patch_num=num_patches), Resize()]
    train = GONRefuge(root=data_root, purpose='train', transform=transforms)
    val = GONRefuge(root=data_root, purpose='val', transform=transforms)

    train_dataloader = DataGeneratorA2(train, batch_size=hparams["batch_size"], flatten_batch=True)
    val_dataloader = DataGeneratorA2(val, batch_size=hparams["batch_size"], flatten_batch=True)

    model = VisionTransformerEmbedded(**{
                                    'embed_dim': 256,
                                    'hidden_dim': 512,
                                    'num_heads': 4,
                                    'num_layers': 2,
                                    'patch_size': patch_size,
                                    'num_patches': num_patches**2,
                                    'num_classes': num_classes,
                                    'dropout': 0.2,
                                    'hparams': hparams
                                })



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patience = 20


    solver = GlaucomaSolver(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        patience=patience
    )

    solver.train()

    os.makedirs('trained_models', exist_ok=True)
    models_path = os.path.join(repo_root, 'trained_models')
    #model.save(os.path.join(models_path, f'vitmodel_batch{hparams["batch_size"]}_lr{hparams["learning_rate"]}_epochs{hparams["epochs"]}.model'))
