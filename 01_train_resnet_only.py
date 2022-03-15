import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from data_processing.cifar_100 import Cifar100
from data_processing.data_generator import DataGenerator
from data_processing.transforms import *
from approach_1.models import BaselineResNet
from approach_1.baseline_training import BaselineSolver



if __name__ == "__main__":
    hparams = {
    'batch_size': 1,
    'learning_rate': 1e-3,
    'epochs': 2,
    'loss_func': torch.nn.CrossEntropyLoss(),
    'optimizer': optim.AdamW
    }

    num_classes = 20
    freeze=False
    patience = 3
    repo_root = os.path.abspath(os.getcwd())
    data_root = os.path.join(repo_root, "data")
    seed = 69
    split = 0.7
    patch_size = 16

    transforms = [RescaleTransform(), ReshapeToTensor(), Patches(patch_size=patch_size), Resize()]
    train = Cifar100(root=data_root, purpose='train', seed=seed, split=0.001, transform=transforms)
    val = Cifar100(root=data_root, purpose='val', seed=seed, split=0.999, transform=transforms)

    train_dataloader = DataGenerator(train, batch_size=hparams['batch_size'])
    val_dataloader = DataGenerator(val, batch_size=hparams['batch_size'])
    model = BaselineResNet(num_classes=num_classes, hparams=hparams, freeze=freeze)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    solver = BaselineSolver(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        patience=patience
    )

    solver.train()

    os.makedirs('trained_models', exist_ok=True)
    models_path = os.path.join(repo_root, 'trained_models')
    model.save(os.path.join(models_path, f'baselinemodel_batch{hparams["batch_size"]}_lr{hparams["learning_rate"]}_epochs{hparams["epochs"]}_freeze{freeze}.model'))

    '''
    plt.title('Loss curves')
    plt.plot(solver.train_loss_history, '-', label='train')
    plt.plot(solver.val_loss_history, '-', label='val')
    plt.legend(loc='lower right')
    plt.xlabel('Iteration')
    plt.show()

    print("Training accuray: %.5f" % (solver.get_dataset_accuracy(dataloaders['train_overfit_single_image'])))
    print("Validation accuray: %.5f" % (solver.get_dataset_accuracy(dataloaders['val_500files'])))
    '''
