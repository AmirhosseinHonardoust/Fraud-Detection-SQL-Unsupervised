#!/usr/bin/env python3
"""Load a transactions CSV into a SQLite database.

Usage:
    python src/create_db.py --csv data/transactions.csv --db fraud.db
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

CREATE_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS transactions ("
    "tx_id INTEGER, user_id TEXT, date TEXT, region TEXT, merchant TEXT, amount REAL)"
)


def load_csv_to_db(csv_path: str | Path, db_path: str | Path) -> Path:
    """Load ``csv_path`` into the ``transactions`` table of ``db_path``.

    Raises FileNotFoundError if the CSV does not exist.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    db_path = Path(db_path)
    df = pd.read_csv(csv_path)

    with sqlite3.connect(db_path) as con:
        con.execute(CREATE_TABLE_SQL)
        # if_exists="replace" drops and recreates the table, so indexes must be
        # (re)created afterwards, not before.
        df.to_sql("transactions", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions (user_id)")
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_transactions_user_date "
            "ON transactions (user_id, date)"
        )

    return db_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load transaction CSV into SQLite")
    parser.add_argument("--csv", required=True, help="Path to the input transactions CSV")
    parser.add_argument("--db", default="fraud.db", help="Path to the output SQLite database")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        db_path = load_csv_to_db(args.csv, args.db)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print("Loaded", args.csv, "->", db_path)


if __name__ == "__main__":
    main()
