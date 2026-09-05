"""Tests for src/utils.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import ensure_outdir, plot_hist, save_csv


def test_ensure_outdir_creates_nested_dirs(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    assert not target.exists()

    result = ensure_outdir(target)

    assert result == target
    assert target.is_dir()


def test_ensure_outdir_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "already_there"
    target.mkdir()

    result = ensure_outdir(target)

    assert result == target
    assert target.is_dir()


def test_save_csv_writes_expected_content(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out_path = tmp_path / "nested" / "out.csv"

    result = save_csv(df, out_path)

    assert result == out_path
    assert out_path.exists()
    loaded = pd.read_csv(out_path)
    pd.testing.assert_frame_equal(loaded, df)


def test_plot_hist_creates_png(tmp_path: Path) -> None:
    series = pd.Series([0.1, 0.2, 0.9, 0.5, 0.3])
    out_path = tmp_path / "charts" / "hist.png"

    result = plot_hist(series, "Test Histogram", out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
