"""Shared matplotlib styling for the paper figures."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

FONT_DIR = Path(__file__).resolve().parent / "fonts"

#: Luminance levels used throughout the paper.
EPSILONS = (0.0, 0.25, 0.5, 0.75)

COLOR_LN = "#eb3920"
COLOR_E_ONLY = "#1f77b4"
COLOR_INORM = "black"
COLOR_NO_NORM = "gray"


def use_paper_style() -> None:
    """Apply the paper's plot defaults, using Arial when it is available."""
    family = "DejaVu Sans"
    for ttf in FONT_DIR.glob("*.ttf"):
        mpl.font_manager.fontManager.addfont(str(ttf))
        family = mpl.font_manager.FontProperties(fname=str(ttf)).get_name()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family, "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 14,
            "axes.labelsize": 20,
            "axes.titlesize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.5,
            "savefig.bbox": "tight",
            "figure.dpi": 110,
        }
    )


def truncate_colormap(cmap, minval=0.2, maxval=0.7, n=256):
    """Sub-range of a colormap, so luminance shading stays legible."""
    return mcolors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )


def luminance_colors(epsilons=EPSILONS, cmap_name="bone", minval=0.2, maxval=0.7):
    """One color per luminance level, dark to light."""
    cmap = truncate_colormap(plt.get_cmap(cmap_name), minval, maxval)
    norm = mcolors.Normalize(vmin=min(epsilons), vmax=max(epsilons))
    return {eps: cmap(norm(eps)) for eps in epsilons}


#: Output format for :func:`save`; set once from the CLI.
FORMAT = "svg"


def set_format(fmt: str) -> None:
    global FORMAT
    FORMAT = fmt


def save(fig, out_dir, name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{FORMAT}"
    fig.savefig(path, format=FORMAT)
    plt.close(fig)
    print(f"wrote {path}")
    return path
