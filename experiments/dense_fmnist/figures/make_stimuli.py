"""Stimulus panel for Figure 1b: the same images under different luminance shifts.

Needs no wandb -- it just samples the augmented dataset.

    python -m experiments.dense_fmnist.figures.make_stimuli --epsilon 0.75
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from experiments.dense_fmnist.figures.style import save, set_format, use_paper_style
from inhibition.data import make_dense_fashion_mnist_dataloaders


def stimulus_grid(data_dir, epsilon, n_images, n_draws, seed):
    """``n_images`` images, each shown under ``n_draws`` independent shifts."""
    torch.manual_seed(seed)
    loader, _ = make_dense_fashion_mnist_dataloaders(
        data_dir, batch_size=n_images, test_batch_size=n_images,
        use_accel=False, brightness_factor=epsilon,
    )
    dataset = loader.dataset
    rows = []
    for draw in range(n_draws):
        rows.append(torch.stack([dataset[i][0].squeeze(0) for i in range(n_images)]))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--epsilon", type=float, default=0.75)
    p.add_argument("--n-images", type=int, default=6)
    p.add_argument("--n-draws", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("figures_out"))
    p.add_argument("--format", default="svg", choices=["svg", "pdf", "png"])
    args = p.parse_args()

    use_paper_style()
    set_format(args.format)
    rows = stimulus_grid(args.data_dir, args.epsilon, args.n_images, args.n_draws, args.seed)

    fig, axes = plt.subplots(
        args.n_draws, args.n_images, figsize=(1.1 * args.n_images, 1.1 * args.n_draws)
    )
    for row_idx, row in enumerate(rows):
        for col_idx in range(args.n_images):
            ax = axes[row_idx, col_idx]
            ax.imshow(row[col_idx], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle(rf"Luminosity range $|\Delta| < \epsilon={args.epsilon}$")
    fig.tight_layout()
    save(fig, args.out, "figure1b")


if __name__ == "__main__":
    main()
