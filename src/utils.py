"""Small filesystem and plotting helpers shared by the pipeline scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Force a non-interactive backend so this works headlessly (CI, servers,
# containers without a display) without extra configuration.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd


def ensure_outdir(path: str | Path) -> Path:
    """Create ``path`` (including parents) if it doesn't exist and return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write ``df`` to ``path`` as CSV, creating parent directories as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def plot_hist(series: pd.Series, title: str, out: str | Path) -> Path:
    """Save a histogram of ``series`` to ``out`` and return the output path."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(series, bins=40)
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Freq")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path
