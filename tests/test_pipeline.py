"""End-to-end tests for the create_db -> detect_fraud_unsupervised pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.create_db import load_csv_to_db
from src.detect_fraud_unsupervised import run_analysis

REPO_ROOT = Path(__file__).resolve().parents[1]
QUERIES_SQL = REPO_ROOT / "src" / "queries.sql"
CREATE_DB_SCRIPT = REPO_ROOT / "src" / "create_db.py"
DETECT_FRAUD_SCRIPT = REPO_ROOT / "src" / "detect_fraud_unsupervised.py"


def _make_synthetic_csv(path: Path, n_rows: int = 200, seed: int = 0) -> Path:
    """Write a small synthetic transactions.csv with the project's schema."""
    rng = pd.Series(range(n_rows))
    df = pd.DataFrame(
        {
            "tx_id": 100000 + rng,
            "user_id": [f"U{(i % 20):04d}" for i in range(n_rows)],
            "date": [f"2024-01-{(i % 28) + 1:02d}" for i in range(n_rows)],
            "region": [["North", "South", "East", "West"][i % 4] for i in range(n_rows)],
            "merchant": [["Grocery", "StoreA", "RideShare"][i % 3] for i in range(n_rows)],
            "amount": [round(10 + (i * 7 % 500) + 0.5, 2) for i in range(n_rows)],
        }
    )
    # Inject a couple of clear outliers so anomaly detection has something to find.
    df.loc[0, "amount"] = 9999.0
    df.loc[1, "amount"] = 8888.0
    df.to_csv(path, index=False)
    return path


def test_load_csv_to_db_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_csv_to_db(tmp_path / "does_not_exist.csv", tmp_path / "out.db")


def test_run_analysis_missing_db_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_analysis(tmp_path / "missing.db", QUERIES_SQL, tmp_path / "outputs")


def test_run_analysis_missing_sql_raises(tmp_path: Path) -> None:
    csv_path = _make_synthetic_csv(tmp_path / "transactions.csv")
    db_path = load_csv_to_db(csv_path, tmp_path / "fraud.db")

    with pytest.raises(FileNotFoundError):
        run_analysis(db_path, tmp_path / "missing.sql", tmp_path / "outputs")


def test_pipeline_end_to_end_produces_expected_artifacts(tmp_path: Path) -> None:
    csv_path = _make_synthetic_csv(tmp_path / "transactions.csv")
    db_path = load_csv_to_db(csv_path, tmp_path / "fraud.db")
    outdir = tmp_path / "outputs"

    run_analysis(db_path, QUERIES_SQL, outdir)

    scores_path = outdir / "fraud_scores.csv"
    summary_path = outdir / "fraud_summary.csv"
    chart_path = outdir / "charts" / "fraud_distribution.png"

    assert scores_path.exists()
    assert summary_path.exists()
    assert chart_path.exists()

    scores = pd.read_csv(scores_path)
    assert list(scores.columns) == ["tx_id", "user_id", "amount", "anomaly_score"]
    assert len(scores) == 200
    assert scores["anomaly_score"].between(0, 1).all()
    # Sorted descending by anomaly score.
    assert (scores["anomaly_score"].diff().dropna() <= 1e-9).all()
    # The injected extreme-amount rows should rank as the most anomalous.
    assert set(scores.head(2)["tx_id"]) == {100000, 100001}

    summary = pd.read_csv(summary_path)
    assert list(summary.columns) == ["user_id", "max_anomaly_score", "total_amount"]
    assert len(summary) > 0


def test_pipeline_is_deterministic(tmp_path: Path) -> None:
    """Recompute the pipeline twice and confirm identical output (fixed random_state)."""
    csv_path = _make_synthetic_csv(tmp_path / "transactions.csv")
    db_path = load_csv_to_db(csv_path, tmp_path / "fraud.db")

    outdir_a = tmp_path / "outputs_a"
    outdir_b = tmp_path / "outputs_b"
    run_analysis(db_path, QUERIES_SQL, outdir_a)
    run_analysis(db_path, QUERIES_SQL, outdir_b)

    scores_a = pd.read_csv(outdir_a / "fraud_scores.csv")
    scores_b = pd.read_csv(outdir_b / "fraud_scores.csv")
    pd.testing.assert_frame_equal(scores_a, scores_b)


def test_run_analysis_rejects_unknown_feature_column(tmp_path: Path) -> None:
    csv_path = _make_synthetic_csv(tmp_path / "transactions.csv")
    db_path = load_csv_to_db(csv_path, tmp_path / "fraud.db")

    with pytest.raises(RuntimeError, match="Feature column"):
        run_analysis(
            db_path,
            QUERIES_SQL,
            tmp_path / "outputs",
            feature_cols=["amount", "not_a_real_column"],
        )


def test_create_db_cli_missing_csv_exits_cleanly(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(CREATE_DB_SCRIPT), "--csv", "does_not_exist.csv", "--db", "x.db"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "Error:" in result.stderr


def test_detect_fraud_cli_missing_db_exits_cleanly(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(DETECT_FRAUD_SCRIPT),
            "--db",
            "missing.db",
            "--sql",
            str(QUERIES_SQL),
            "--outdir",
            str(tmp_path / "outputs"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "Error:" in result.stderr


def test_detect_fraud_cli_missing_sql_exits_cleanly(tmp_path: Path) -> None:
    csv_path = _make_synthetic_csv(tmp_path / "transactions.csv")
    db_path = load_csv_to_db(csv_path, tmp_path / "fraud.db")

    result = subprocess.run(
        [
            sys.executable,
            str(DETECT_FRAUD_SCRIPT),
            "--db",
            str(db_path),
            "--sql",
            "missing.sql",
            "--outdir",
            str(tmp_path / "outputs"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "Error:" in result.stderr
