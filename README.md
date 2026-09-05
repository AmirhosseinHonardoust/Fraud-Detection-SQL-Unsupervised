<div align="center">

# Fraud Detection, SQL + Unsupervised ML

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Isolation%20Forest-orange)
![SQLite](https://img.shields.io/badge/SQLite-Feature%20Engineering-lightgrey)
![Status](https://img.shields.io/badge/Status-Educational%20ML%20Project-purple)
[![CI](https://github.com/AmirhosseinHonardoust/Fraud-Detection-SQL-Unsupervised/actions/workflows/ci.yml/badge.svg)](https://github.com/AmirhosseinHonardoust/Fraud-Detection-SQL-Unsupervised/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

A practical machine learning project that turns raw bank transactions into a **ranked anomaly signal**, using **SQL (SQLite) feature engineering** combined with an **unsupervised Isolation Forest** pipeline, a fully typed and tested codebase, and command-line tooling for reproducible runs.

> **Important:** This project is an **unsupervised anomaly-ranking demo**, not a verified fraud-detection system.
>
> The model has never seen a labeled fraud/not-fraud example. It ranks transactions by how statistically unusual their aggregated behavior looks, relative to other transactions in the same dataset. A high `anomaly_score` means "unusual," not "confirmed fraudulent," and should not be used on its own for account actions, blocking, or any high-stakes decision.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What This Project Does](#what-this-project-does)
- [What This Project Does Not Do](#what-this-project-does-not-do)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Schema](#dataset-schema)
- [SQL Feature Engineering](#sql-feature-engineering)
- [Model Output](#model-output)
- [Visual Reports](#visual-reports)
- [Testing and CI](#testing-and-ci)
- [Code Quality](#code-quality)
- [Limitations](#limitations)
- [Responsible Use](#responsible-use)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Project Overview

Fraud detection is often presented as if a model can simply "detect fraud." In reality, most real-world transaction data has no fraud labels at all, so a supervised classifier isn't even an option, and an anomaly score is only useful if it can support a defensible next step:

- rank transactions by how unusual they look, not classify them as guilty
- surface the aggregated behavioral features driving that ranking
- stay deterministic and reproducible across runs
- make the model's tunables (feature set, contamination, seed) explicit rather than hidden

This project demonstrates an end-to-end, honestly-scoped unsupervised workflow on transaction data: SQL-based feature engineering, an Isolation Forest anomaly model, normalized scoring, ranked CSV/summary output, a distribution chart, and a fully typed, tested, CI-gated codebase.

The goal is to show how SQL and Python combine into a **reproducible anomaly-ranking pipeline**, not to claim a production fraud-detection accuracy number that unlabeled data cannot support.

---

## What This Project Does

This project can:

- Load a transactions CSV into a SQLite database with query-friendly indexes
- Compute user-level and daily behavioral aggregates entirely in SQL
- Feed those aggregates into an unsupervised Isolation Forest
- Produce a normalized `anomaly_score` (0–1, higher = more unusual) per transaction
- Rank transactions and summarize the most anomalous users
- Generate an anomaly-score distribution chart
- Run deterministically given a fixed random seed
- Accept a configurable feature set, contamination rate, and seed via CLI flags
- Run automated tests and a GitHub Actions CI quality gate on every push/PR

---

## What This Project Does Not Do

This project does **not**:

- Prove that any transaction is fraudulent
- Train on or evaluate against labeled fraud data (none exists in this dataset)
- Report precision, recall, or accuracy against ground truth
- Detect all types of financial crime or adversarial behavior
- Guarantee real-world performance on production transaction volumes or patterns
- Make automated blocking, freezing, or reporting decisions

A production fraud system would need labeled outcomes, a feedback loop, case-management workflows, regulatory review, and human analysts in the loop.

---

## Key Features

- **SQL-first feature engineering**, user-level and daily aggregates computed as SQLite views, not in Python
- **Quote-aware SQL statement splitting** so multi-statement `.sql` files (including string literals containing `;`) are parsed correctly
- **Isolation Forest** unsupervised anomaly model with configurable `--contamination` and `--random-state`
- **Configurable feature set** via `--feature-cols`, without editing code
- **Deterministic pipeline**, identical inputs and seed always produce identical output (tested)
- **Ranked CSV output** plus a per-user summary table
- **Anomaly-score distribution chart** via matplotlib (headless-safe, `Agg` backend)
- **100% test coverage on `src/`**, enforced in CI
- **Ruff, Black, and mypy** quality gate on every push/PR, across Python 3.11 and 3.12
- **Pre-commit hooks** and a `Makefile` so the CI gate can be run identically and locally

---

## System Workflow

```text
Raw transactions CSV
        ↓
SQLite ingestion + indexing (create_db.py)
        ↓
SQL feature engineering (queries.sql: user + daily aggregates)
        ↓
Isolation Forest (unsupervised, scikit-learn)
        ↓
Min-max normalized anomaly_score (0–1)
        ↓
Ranked scores + per-user summary
        ↓
Distribution chart + CSV review
```

---

## Project Structure

```text
Fraud-Detection-SQL-Unsupervised/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   └── transactions.csv
│
├── outputs/
│   ├── charts/
│   │   └── fraud_distribution.png
│   ├── fraud_scores.csv
│   └── fraud_summary.csv
│
├── src/
│   ├── __init__.py
│   ├── create_db.py
│   ├── detect_fraud_unsupervised.py
│   ├── queries.sql
│   └── utils.py
│
├── tests/
│   ├── test_pipeline.py
│   └── test_utils.py
│
├── .pre-commit-config.yaml
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── README.md
├── conftest.py
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── requirements-lock.txt
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmirhosseinHonardoust/Fraud-Detection-SQL-Unsupervised.git
cd Fraud-Detection-SQL-Unsupervised
```

### 2. Create a Virtual Environment

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

For development tools (Ruff, Black, mypy, pytest, coverage):

```bash
pip install -r requirements-dev.txt
pre-commit install   # optional: runs the quality gate on every commit
```

For an exact, fully-resolved environment (pinned transitive dependencies too):

```bash
pip install -r requirements-lock.txt
```

---

## Quick Start

Load transactions into SQLite:

```bash
python src/create_db.py --csv data/transactions.csv --db fraud.db
```

Run anomaly detection:

```bash
python src/detect_fraud_unsupervised.py --db fraud.db --sql src/queries.sql --outdir outputs
```

Optional flags tune the pipeline without editing code:

```bash
python src/detect_fraud_unsupervised.py \
  --db fraud.db \
  --sql src/queries.sql \
  --outdir outputs \
  --feature-cols amount,tx_count,avg_amount,total_amount,daily_tx,daily_amount \
  --contamination 0.02 \
  --random-state 7
```

`--feature-cols` (comma-separated) selects which SQL result columns feed the Isolation Forest; `--contamination` sets the expected anomaly proportion; `--random-state` fixes the seed for reproducibility.

---

## Dataset Schema
<div align="center">

| Column | Description |
|---|---|
| `tx_id` | Transaction ID |
| `user_id` | Unique user identifier |
| `date` | Transaction date |
| `region` | User region |
| `merchant` | Merchant name or type |
| `amount` | Transaction amount |
</div>

---

## SQL Feature Engineering

Feature generation happens entirely in `src/queries.sql`, executed against the SQLite database via `run_analysis()`. It builds two temporary views, one for user-level aggregates, one for daily activity, then joins both back onto every transaction row:

```sql
CREATE TEMP VIEW user_stats AS
SELECT user_id, COUNT(*) AS tx_count, AVG(amount) AS avg_amount, SUM(amount) AS total_amount
FROM transactions
GROUP BY user_id;

CREATE TEMP VIEW daily_user AS
SELECT user_id, date, COUNT(*) AS daily_tx, SUM(amount) AS daily_amount
FROM transactions
GROUP BY user_id, date;

SELECT t.tx_id, t.user_id, t.date, t.region, t.merchant, t.amount,
       us.tx_count, us.avg_amount, us.total_amount,
       COALESCE(du.daily_tx, 0) AS daily_tx,
       COALESCE(du.daily_amount, 0.0) AS daily_amount
FROM transactions t
LEFT JOIN user_stats us ON t.user_id = us.user_id
LEFT JOIN daily_user du ON t.user_id = du.user_id AND t.date = du.date;
```

> Reformatted here for readability, the bundled file uses implicit column aliases (e.g. `COUNT(*) tx_count`); the logic is identical. The statement splitter that parses this file (`src/utils.py::split_sql_statements`) is quote-aware, so a `;` inside a string literal elsewhere in a custom `.sql` file won't be mistaken for a statement boundary.

---

## Model Output

The pipeline produces a continuous score, not a label:
<div align="center">

| Column | Meaning |
|---|---|
| `anomaly_score` | 0–1, higher = more statistically unusual relative to the rest of the dataset |
</div>

```text
outputs/fraud_scores.csv      every transaction, ranked by anomaly_score
outputs/fraud_summary.csv     per-user max anomaly_score and total_amount
outputs/charts/fraud_distribution.png
```

`fraud_scores.csv` is sorted descending by `anomaly_score`; `fraud_summary.csv` groups the top 1,000 most anomalous transactions by `user_id`. There is no fixed threshold that converts a score into a `FRAUD` / `NOT_FRAUD` label, `--contamination` only tells Isolation Forest what proportion of the data to *treat* as the anomalous tail internally, it doesn't create a hard cutoff in the output.

---

## Visual Reports

### Anomaly Score Distribution

<div align="center">
<img width="1200" height="750" alt="Anomaly score distribution" src="https://github.com/user-attachments/assets/8adcb416-c0b9-4e3c-8d60-34f11983b3eb" />
</div>

**Analysis:** Most transactions cluster at low anomaly scores; the right-side tail is the small set of transactions Isolation Forest found hardest to explain with the same behavioral pattern as everyone else. That tail is the review queue, not a fraud list.

---

## Testing and CI

Run unit tests locally:

```bash
pytest
```

Run the full quality gate (same checks CI runs, on Python 3.11 and 3.12):

```bash
make gate       # lint + format-check + typecheck + test + coverage
```

or run any step on its own:

```bash
make lint           # ruff check src tests
make format-check   # black --check src tests
make typecheck      # mypy src tests
make test           # pytest
make coverage       # pytest --cov=src --cov-report=term-missing
```

The GitHub Actions workflow checks, on a 3.11 / 3.12 matrix:

- dependency installation
- linting with Ruff
- format checking with Black
- type checking with mypy (`src` and `tests`)
- the full test suite with coverage (95% minimum on `src/`, currently 100%)

CI is defined in:

```text
.github/workflows/ci.yml
```

---

## Code Quality

The project separates responsibilities across a small number of modules:
<div align="center">
        
| Module | Purpose |
|---|---|
| `src/create_db.py` | Loads a transactions CSV into SQLite and creates query indexes |
| `src/queries.sql` | SQL views and final SELECT that produce the feature table |
| `src/detect_fraud_unsupervised.py` | Runs the SQL, fits Isolation Forest, normalizes scores, writes artifacts |
| `src/utils.py` | Filesystem helpers, plotting, and the quote-aware SQL statement splitter |
</div>

Tooling is configured through `pyproject.toml` (Ruff, Black, mypy, pytest, coverage), `Makefile`, and `.pre-commit-config.yaml`. `src/` is also an installable package (`pip install -e .`).

---

## Limitations

This project has important limitations:

- `data/transactions.csv` is a single, static, synthetic-style dataset (~3 MB); there is no live or streaming ingestion path
- There are no fraud labels anywhere in this dataset, so no precision, recall, or accuracy figure can be honestly reported
- `anomaly_score` is a relative ranking within one run's dataset, not a calibrated probability of fraud
- The feature set, contamination rate, and seed are the only tunables, there is no config file or hyperparameter search
- `src/queries.sql`'s final statement is assumed to be the feature-producing `SELECT`; the pipeline validates the file isn't empty but not that it follows this convention

The project is strongest as a portfolio demonstration of a clean, reproducible, well-tested SQL + unsupervised-ML pipeline, not as a fraud-detection accuracy benchmark.

---

## Responsible Use

This repository is intended for:

- data engineering and machine learning education
- demonstrating SQL-based feature engineering paired with unsupervised ML
- practicing reproducible, CI-gated Python pipeline design
- portfolio demonstration

It should not be used as-is for:

- real account, transaction, or fraud-case decisions
- automated blocking, freezing, or reporting of users
- any high-stakes financial or legal determination

Any real deployment would require labeled outcome data, a human review workflow, model monitoring, and regulatory/compliance sign-off.

---

## Future Improvements

Potential next improvements:

- Add a synthetic labeled evaluation set to sanity-check ranking quality against known injected anomalies
- Support additional unsupervised models (e.g. Local Outlier Factor, autoencoder) behind the same CLI
- Add a lightweight dashboard for interactive review of `fraud_summary.csv`
- Extend feature engineering with velocity/z-score features beyond the current aggregates
- Add Docker support for a fully reproducible run environment
- Add calibration/stability metrics across repeated runs with different seeds

---

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- SQLite
- pytest
- Ruff
- Black
- mypy
- pre-commit
- GitHub Actions

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

MIT, see [LICENSE](LICENSE).

This project is intended for educational, research, and portfolio purposes. If you use or modify it, please keep the responsible-use notes and limitations clear.
