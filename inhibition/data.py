from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from torchvision import datasets, transforms


class RandomAdjustBrightness:
    def __init__(self, brightness_factor: float):
        self.brightness_factor = brightness_factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        random_adjustment = (np.random.rand() * 2 - 1) * self.brightness_factor
        x = x + random_adjustment
        return torch.clamp(x, 0, 1)


def default_mnist_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def make_mnist_dataloaders(
    data_dir: str | Path,
    *,
    batch_size: int,
    test_batch_size: int,
    use_accel: bool,
    download: bool = True,
):
    data_dir = Path(data_dir)
    transform = default_mnist_transform()
    train_set = datasets.MNIST(
        str(data_dir), train=True, download=download, transform=transform
    )
    test_set = datasets.MNIST(str(data_dir), train=False, transform=transform)

    train_kwargs: dict = {"batch_size": batch_size}
    test_kwargs: dict = {"batch_size": test_batch_size}
    if use_accel:
        accel_kwargs = {
            "num_workers": 1,
            "persistent_workers": True,
            "pin_memory": True,
            "shuffle": True,
        }
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)
    return train_loader, test_loader


def fashion_mnist_normalize():
    """Channel stats for Fashion-MNIST (grayscale)."""
    return transforms.Normalize((0.2860,), (0.3530,))


def fashion_mnist_transform(brightness_factor: float):
    """ToTensor, random brightness jitter, then normalize (train and eval)."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            RandomAdjustBrightness(brightness_factor),
            fashion_mnist_normalize(),
        ]
    )


def make_fashion_mnist_dataloaders(
    data_dir: str | Path,
    *,
    batch_size: int,
    test_batch_size: int,
    use_accel: bool,
    brightness_factor: float = 0.1,
    download: bool = True,
):
    data_dir = Path(data_dir)
    transform = fashion_mnist_transform(brightness_factor)
    train_set = datasets.FashionMNIST(
        str(data_dir), train=True, download=download, transform=transform
    )
    test_set = datasets.FashionMNIST(
        str(data_dir), train=False, transform=transform
    )

    train_kwargs: dict = {"batch_size": batch_size}
    test_kwargs: dict = {"batch_size": test_batch_size}
    if use_accel:
        accel_kwargs = {
            "num_workers": 1,
            "persistent_workers": True,
            "pin_memory": True,
            "shuffle": True,
        }
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)
    return train_loader, test_loader
