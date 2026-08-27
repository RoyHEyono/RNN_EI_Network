from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from torchvision import datasets, transforms


class RandomAdjustBrightness:
    """Luminance shift of the paper: one scalar per image, then clamp to [0, 1].

    ``fixed=True`` applies ``brightness_factor`` itself rather than a draw from
    ``Uniform(-eps, +eps)``, for evaluating at a single known luminance.
    """

    def __init__(self, brightness_factor: float, fixed: bool = False):
        self.brightness_factor = brightness_factor
        self.fixed = fixed

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.fixed:
            random_adjustment = self.brightness_factor
        else:
            random_adjustment = (np.random.rand() * 2 - 1) * self.brightness_factor
        x = x + random_adjustment
        return torch.clamp(x, 0, 1)


class RandomAdjustContrast:
    """Contrast scale around the per-image mean: ``c ~ Unif[1-γ, 1+γ]`` (clamped ≥ 0).

    ``fixed=True`` applies ``c = 1 + gamma`` instead of a random draw, for
    evaluating at a single known contrast (mirrors :class:`RandomAdjustBrightness`).
    ``gamma=0`` is identity.
    """

    def __init__(self, mode: float, fixed: bool = False):
        self.gamma = mode
        self.fixed = fixed

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.gamma == 0 and not self.fixed:
            return x

        if self.fixed:
            c = max(0.0, 1.0 + self.gamma)
        else:
            c = max(0.0, 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * self.gamma)
        mu_img = x.mean(dim=(-2, -1), keepdim=True)
        x_out = (x - mu_img) * c + mu_img
        return torch.clamp(x_out, 0.0, 1.0)


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


def fashion_mnist_transform(brightness_factor: float, contrast_factor: float = 0.0):
    """ToTensor, contrast then brightness jitter, then normalize (train and eval)."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            RandomAdjustContrast(contrast_factor),
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
    contrast_factor: float = 0.0,
    download: bool = True,
):
    data_dir = Path(data_dir)
    transform = fashion_mnist_transform(brightness_factor, contrast_factor)
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


def dense_fashion_mnist_transform(
    epsilon: float,
    fixed: bool = False,
    *,
    jitter: str = "brightness",
):
    """Raw ``[0, 1]`` pixels plus luminance or contrast jitter for dense I-Norm runs.

    Deliberately omits the mean/std ``Normalize`` used by
    :func:`fashion_mnist_transform`: the paper defines epsilon relative to the
    ``[0, 1]`` pixel range, so re-standardizing afterwards would rescale it.

    ``jitter`` is ``"brightness"`` (default, paper luminance shift) or
    ``"contrast"`` (:class:`RandomAdjustContrast` with the same epsilon).
    """
    if jitter == "brightness":
        adjust = RandomAdjustBrightness(epsilon, fixed=fixed)
    elif jitter == "contrast":
        adjust = RandomAdjustContrast(epsilon, fixed=fixed)
    else:
        raise ValueError(f"jitter must be 'brightness' or 'contrast', got {jitter!r}")
    return transforms.Compose(
        [
            transforms.ToTensor(),
            adjust,
        ]
    )


def make_dense_fashion_mnist_dataloaders(
    data_dir: str | Path,
    *,
    batch_size: int,
    test_batch_size: int,
    use_accel: bool,
    brightness_factor: float = 0.0,
    brightness_factor_eval: float = 0.0,
    download: bool = True,
    num_workers: int = 2,
    dataset: str = "fashionmnist",
):
    """Fashion-MNIST loaders for the dense I-Norm experiments.

    ``dataset``:
      * ``fashionmnist`` — random luminance jitter (paper default)
      * ``fashionmnist_contrast`` — random contrast jitter with the same epsilon

    Train and test see the same random jitter unless
    ``brightness_factor_eval`` is non-zero, in which case the test set is shifted
    by that fixed amount instead. The CLI still names this flag
    ``brightness_factor`` / ``brightness_factor_eval`` for both variants.
    """
    if dataset == "fashionmnist":
        jitter = "brightness"
    elif dataset == "fashionmnist_contrast":
        jitter = "contrast"
    else:
        raise ValueError(
            f"dataset must be 'fashionmnist' or 'fashionmnist_contrast', got {dataset!r}"
        )

    data_dir = Path(data_dir)
    train_transform = dense_fashion_mnist_transform(
        brightness_factor, jitter=jitter
    )
    if brightness_factor_eval:
        test_transform = dense_fashion_mnist_transform(
            brightness_factor_eval, fixed=True, jitter=jitter
        )
    else:
        test_transform = train_transform

    train_set = datasets.FashionMNIST(
        str(data_dir), train=True, download=download, transform=train_transform
    )
    test_set = datasets.FashionMNIST(
        str(data_dir), train=False, download=download, transform=test_transform
    )

    train_kwargs: dict = {"batch_size": batch_size, "shuffle": True}
    test_kwargs: dict = {"batch_size": test_batch_size, "shuffle": False}
    if use_accel:
        accel_kwargs = {"num_workers": num_workers, "pin_memory": True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)
    return train_loader, test_loader
