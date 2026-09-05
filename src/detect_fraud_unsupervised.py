#!/usr/bin/env python3
"""Unsupervised fraud detection: SQL feature engineering + Isolation Forest.

Usage:
    python src/detect_fraud_unsupervised.py --db fraud.db --sql src/queries.sql --outdir outputs
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

try:
    # Direct execution (``python src/detect_fraud_unsupervised.py``, as documented
    # in the README): the script's own directory is on sys.path, so this works.
    from utils import ensure_outdir, plot_hist, save_csv, split_sql_statements
except ImportError:
    # Imported as part of the ``src`` package (e.g. from tests).
    from src.utils import ensure_outdir, plot_hist, save_csv, split_sql_statements

DEFAULT_FEATURE_COLS = [
    "amount",
    "tx_count",
    "avg_amount",
    "total_amount",
    "daily_tx",
    "daily_amount",
]


def run_analysis(
    db_path: str | Path,
    sql_path: str | Path,
    outdir: str | Path,
    feature_cols: list[str] | None = None,
    contamination: float = 0.02,
    random_state: int = 7,
) -> None:
    """Run the SQL feature engineering + Isolation Forest pipeline.

    Args:
        db_path: Path to the SQLite database (created via ``create_db.py``).
        sql_path: Path to the SQL file with feature-engineering statements. All
            statements except the last are executed as setup (e.g. CREATE VIEW);
            the last statement must be the SELECT that produces the feature table.
        outdir: Directory where ``fraud_scores.csv``, ``fraud_summary.csv``, and
            ``charts/fraud_distribution.png`` are written.
        feature_cols: Columns used as Isolation Forest input features. Defaults
            to ``DEFAULT_FEATURE_COLS``, which matches the columns produced by
            the bundled ``src/queries.sql``.
        contamination: Expected proportion of anomalies, passed to
            ``IsolationForest``. Defaults to 0.02 (~2% anomalies).
        random_state: Random seed passed to ``IsolationForest``. Defaults to 7;
            fixing it keeps the pipeline deterministic across runs.

    Raises:
        FileNotFoundError: If ``db_path`` or ``sql_path`` do not exist.
        RuntimeError: If the SQL file has no statements, or the final SELECT
            returns no rows.
    """
    db_path = Path(db_path)
    sql_path = Path(sql_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    feature_cols = feature_cols or DEFAULT_FEATURE_COLS

    outdir = ensure_outdir(outdir)
    charts_dir = ensure_outdir(Path(outdir) / "charts")

    # Read the SQL file and split into statements (quote-aware: semicolons inside
    # string/identifier literals do not split a statement).
    with open(sql_path, encoding="utf-8") as f:
        sql_text = f.read()
    statements = split_sql_statements(sql_text)
    if not statements:
        raise RuntimeError("No SQL statements found in the provided file.")

    # Everything except the last statement are setup statements (e.g., CREATE VIEWs).
    setup_script = ";\n".join(statements[:-1]) + (";" if len(statements) > 1 else "")
    final_select = statements[-1]

    # Execute SQL: first the setup (if any), then the final SELECT.
    with sqlite3.connect(db_path) as con:
        if setup_script:
            con.executescript(setup_script)
        df = pd.read_sql_query(final_select, con)

    if df.empty:
        raise RuntimeError("Final SELECT returned no rows. Check your data and SQL.")

    # Features for anomaly detection
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Feature column(s) not found in SQL result: {missing}")
    X = df[feature_cols].fillna(0)

    # Isolation Forest (unsupervised)
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )
    # decision_function: higher = more normal. Convert to [0,1] anomaly score,
    # where higher = more anomalous, via min-max normalization.
    decision = model.fit(X).decision_function(X)
    d_min, d_max = decision.min(), decision.max()
    df["anomaly_score"] = (d_max - decision) / (d_max - d_min + 1e-9)

    # Rank by anomaly score
    df_sorted = df.sort_values("anomaly_score", ascending=False)

    # Save artifacts
    save_csv(
        df_sorted[["tx_id", "user_id", "amount", "anomaly_score"]],
        Path(outdir) / "fraud_scores.csv",
    )
    top_summary = (
        df_sorted.head(1000)[["user_id", "amount", "anomaly_score"]]
        .groupby("user_id")
        .agg(max_anomaly_score=("anomaly_score", "max"), total_amount=("amount", "sum"))
        .reset_index()
        .sort_values(["max_anomaly_score", "total_amount"], ascending=False)
    )
    save_csv(top_summary, Path(outdir) / "fraud_summary.csv")

    # Plot histogram of anomaly scores
    plot_hist(
        df_sorted["anomaly_score"],
        "Anomaly Score Distribution",
        charts_dir / "fraud_distribution.png",
    )

    print("Artifacts saved to:", str(Path(outdir).resolve()))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Unsupervised fraud detection (Isolation Forest) with SQL features"
    )
    ap.add_argument("--db", default="fraud.db", help="Path to SQLite database")
    ap.add_argument(
        "--sql", default="src/queries.sql", help="Path to SQL file (feature engineering)"
    )
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    ap.add_argument(
        "--feature-cols",
        default=None,
        help=(
            "Comma-separated list of columns from the SQL result to use as "
            "Isolation Forest input features. Defaults to the columns produced "
            "by the bundled src/queries.sql "
            f"({','.join(DEFAULT_FEATURE_COLS)})."
        ),
    )
    ap.add_argument(
        "--contamination",
        type=float,
        default=0.02,
        help="Expected proportion of anomalies (IsolationForest contamination). Default: 0.02",
    )
    ap.add_argument(
        "--random-state",
        type=int,
        default=7,
        help="Random seed for IsolationForest, for reproducible results. Default: 7",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    feature_cols = (
        [c.strip() for c in args.feature_cols.split(",") if c.strip()]
        if args.feature_cols
        else None
    )
    try:
        run_analysis(
            args.db,
            args.sql,
            args.outdir,
            feature_cols=feature_cols,
            contamination=args.contamination,
            random_state=args.random_state,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
