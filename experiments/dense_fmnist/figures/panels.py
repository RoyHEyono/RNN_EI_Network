"""Reusable panel primitives shared by the figure scripts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

from experiments.dense_fmnist.figures.style import luminance_colors


def significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def scatter_vs_diagonal(ax, pairs_by_eps, xlabel, ylabel, lims=None, cmap_name="bone"):
    """Paired accuracies against the identity line, one color per luminance.

    Points below the diagonal mean the x-axis condition did better.
    """
    colors = luminance_colors(tuple(pairs_by_eps), cmap_name=cmap_name)
    all_values = [v for pairs in pairs_by_eps.values() for pair in pairs for v in pair]
    if lims is None:
        if not all_values:
            lims = (0, 100)
        else:
            lo, hi = min(all_values), max(all_values)
            pad = max(0.5, 0.05 * (hi - lo))
            lims = (lo - pad, hi + pad)

    ax.plot(lims, lims, color="black", linestyle="--", zorder=0)
    for eps, pairs in pairs_by_eps.items():
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        ax.scatter(xs, ys, color=colors[eps], label=rf"$\epsilon$={eps}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.legend(loc="upper left", title="Luminosity", title_fontsize=12)
    return ax


def boxplot_with_significance(
    ax,
    groups,
    labels,
    *,
    baseline=None,
    baseline_label=r"LN$^+$",
    ylabel="Test Acc",
    facecolors=None,
    drop_below=10.0,
):
    """Boxplots compared against a dashed baseline, with Mann-Whitney stars.

    Runs that collapsed to near chance are dropped (and marked with a red tilde)
    so a handful of diverged hyperparameters do not flatten the y-scale -- the
    same convention the paper uses.
    """
    cleaned, collapsed = [], []
    for values in groups:
        values = [v for v in values if v is not None]
        keep = [v for v in values if v > drop_below]
        # Only drop outliers when the group is otherwise healthy.
        if keep and np.median(keep) >= 50.0 and len(keep) < len(values):
            collapsed.append(True)
            cleaned.append(keep)
        else:
            collapsed.append(False)
            cleaned.append(values)

    positions = np.arange(1, len(cleaned) + 1)
    box = ax.boxplot(cleaned, positions=positions, patch_artist=True,
                     widths=0.55, showfliers=False)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    if facecolors:
        for patch, color in zip(box["boxes"], facecolors):
            patch.set(facecolor=color)

    if baseline:
        base = [v for v in baseline if v is not None]
        ax.axhline(float(np.median(base)), color="red", linestyle=":", lw=2)
        ax.text(0.99, float(np.median(base)), baseline_label, color="red",
                ha="right", va="bottom", transform=ax.get_yaxis_transform(),
                fontsize=13)
        span = ax.get_ylim()
        head = span[1]
        for pos, values, has_collapsed in zip(positions, cleaned, collapsed):
            if not values:
                continue
            _, p = mannwhitneyu(values, base, alternative="two-sided")
            ax.text(pos, head, significance_stars(p), ha="center", va="bottom",
                    fontsize=14)
            if has_collapsed:
                ax.text(pos, span[0], "~", color="red", ha="center", va="top",
                        fontsize=16)
        ax.set_ylim(span[0], head + 0.08 * (head - span[0]))

    ax.set_ylabel(ylabel)
    return ax


def two_group_boxplot(ax, values_a, values_b, label_a, label_b, colors, ylabel="Accuracy"):
    """Two boxes with a significance bracket between them (Fig. 2b)."""
    box = ax.boxplot([values_a, values_b], patch_artist=True, widths=0.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([label_a, label_b])
    for patch, color in zip(box["boxes"], colors):
        patch.set(facecolor=color)

    _, p = mannwhitneyu(values_a, values_b, alternative="two-sided")
    y_max = max(max(values_a), max(values_b))
    h = 0.02 * (y_max - min(min(values_a), min(values_b)) + 1)
    ax.plot([1, 1, 2, 2], [y_max, y_max + h, y_max + h, y_max], lw=1.5, c="black")
    ax.text(1.5, y_max + h * 1.1, significance_stars(p), ha="center", va="bottom",
            fontsize=18)
    ax.set_ylabel(ylabel)
    return p


def alignment_histograms(values_by_panel, xlabel, *, bins=10, xlim=(0, 1),
                         color="0.3", figsize=(7, 5.5)):
    """Stacked histograms, one per entry of ``values_by_panel`` (title -> values)."""
    n = len(values_by_panel)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (title, values) in zip(axes, values_by_panel.items()):
        ax.hist([v for v in values if v is not None], bins=bins, range=xlim,
                color=color, edgecolor="white")
        ax.set_ylabel("Count")
        ax.set_title(title, loc="left", fontsize=14)
        ax.set_xlim(*xlim)
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    return fig, axes
