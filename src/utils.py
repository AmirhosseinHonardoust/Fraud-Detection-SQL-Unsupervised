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


def split_sql_statements(sql_text: str) -> list[str]:
    """Split ``sql_text`` into individual statements on top-level ``;`` characters.

    Unlike a naive ``sql_text.split(";")``, this ignores semicolons that appear
    inside single-quoted ``'...'`` or double-quoted ``"..."`` SQL string/identifier
    literals (including the standard ``''``/``""`` escaped-quote convention), so a
    literal like ``'a;b'`` is not mistaken for a statement boundary. Empty
    statements (blank lines, trailing whitespace) are dropped.
    """
    statements = []
    buf: list[str] = []
    quote_char: str | None = None
    i = 0
    n = len(sql_text)
    while i < n:
        ch = sql_text[i]
        if quote_char is not None:
            buf.append(ch)
            if ch == quote_char:
                # A doubled quote ('' or "") is an escaped quote, not the closer.
                if i + 1 < n and sql_text[i + 1] == quote_char:
                    buf.append(sql_text[i + 1])
                    i += 1
                else:
                    quote_char = None
        elif ch in ("'", '"'):
            quote_char = ch
            buf.append(ch)
        elif ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
        else:
            buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


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
