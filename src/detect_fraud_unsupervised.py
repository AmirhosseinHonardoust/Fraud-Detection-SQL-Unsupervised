#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

from utils import ensure_outdir, save_csv, plot_hist


def run_analysis(db_path: str | Path, sql_path: str | Path, outdir: str | Path) -> None:
    outdir = ensure_outdir(outdir)
    charts_dir = ensure_outdir(Path(outdir) / "charts")

    # Read the SQL file and split into statements.
    with open(sql_path, "r", encoding="utf-8") as f:
        sql_text = f.read()
    statements = [s.strip() for s in sql_text.split(";") if s.strip()]
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
    feature_cols = ["amount", "tx_count", "avg_amount", "total_amount", "daily_tx", "daily_amount"]
    X = df[feature_cols].fillna(0)

    # Isolation Forest (unsupervised)
    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,   # ~2% anomalies
        random_state=7,
    )
    # decision_function: higher = more normal → convert to [0,1] anomaly score where higher = more anomalous
    decision = model.fit(X).decision_function(X)
    df["anomaly_score"] = (-decision - decision.min()) / (decision.max() - decision.min() + 1e-9)

    # Rank by anomaly score
    df_sorted = df.sort_values("anomaly_score", ascending=False)

    # Save artifacts
    save_csv(df_sorted[["tx_id", "user_id", "amount", "anomaly_score"]], Path(outdir) / "fraud_scores.csv")
    top_summary = (
        df_sorted.head(1000)[["user_id", "amount", "anomaly_score"]]
        .groupby("user_id")
        .agg(max_anomaly_score=("anomaly_score", "max"), total_amount=("amount", "sum"))
        .reset_index()
        .sort_values(["max_anomaly_score", "total_amount"], ascending=False)
    )
    save_csv(top_summary, Path(outdir) / "fraud_summary.csv")

    # Plot histogram of anomaly scores
    plot_hist(df_sorted["anomaly_score"], "Anomaly Score Distribution", charts_dir / "fraud_distribution.png")

    print("Artifacts saved to:", str(Path(outdir).resolve()))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Unsupervised fraud detection (Isolation Forest) with SQL features")
    ap.add_argument("--db", default="fraud.db", help="Path to SQLite database")
    ap.add_argument("--sql", default="src/queries.sql", help="Path to SQL file (feature engineering)")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(args.db, args.sql, args.outdir)


if __name__ == "__main__":
    main()
