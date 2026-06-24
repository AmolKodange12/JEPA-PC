"""MNIST loading utilities for Stage 1."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_mnist(data_dir: str = "data/", batch_size: int = 64, flatten: bool = True):
    """Return (train_loader, test_loader) for MNIST.

    Images are normalised to [0, 1] and optionally flattened to (784,).
    Labels are kept as class indices (int64).
    """
    tf_list = [transforms.ToTensor()]
    if flatten:
        tf_list.append(transforms.Lambda(lambda x: x.view(-1)))
    tf = transforms.Compose(tf_list)

    train_ds = datasets.MNIST(root=data_dir, train=True,  download=False, transform=tf)
    test_ds  = datasets.MNIST(root=data_dir, train=False, download=False, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader
