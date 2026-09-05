# Fraud Detection  [SQL + Python (Unsupervised)]

[![CI](https://github.com/AmirhosseinHonardoust/Fraud-Detection-SQL-Unsupervised/actions/workflows/ci.yml/badge.svg)](https://github.com/AmirhosseinHonardoust/Fraud-Detection-SQL-Unsupervised/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Detect potentially fraudulent bank transactions using **SQL (SQLite)** for feature engineering and **Python** for unsupervised anomaly detection with Isolation Forest.

---

## Overview

This project demonstrates a practical fraud detection workflow where no labeled data is available.  
It integrates **SQL-based data aggregation** with **machine learning anomaly detection**, showing how data engineers and analysts can uncover unusual transaction patterns in banking or financial systems.

---

## Workflow

1. **Load transaction data into SQLite**
2. **Run SQL feature engineering**
   - Compute user-level metrics (average amount, total amount, number of transactions)
   - Compute daily activity (daily totals and transaction counts)
3. **Apply Isolation Forest** to detect anomalies based on aggregated behavioral features
4. **Generate outputs**
   - Ranked anomaly scores
   - Summary tables
   - Distribution chart

---

## Project Structure

```
fraud-detection-sql-unsupervised/
├─ README.md
├─ requirements.txt
├─ data/
│  └─ transactions.csv
├─ src/
│  ├─ create_db.py
│  ├─ queries.sql
│  ├─ detect_fraud_unsupervised.py
│  └─ utils.py
└─ outputs/
   ├─ fraud_scores.csv
   ├─ fraud_summary.csv
   └─ charts/
       └─ fraud_distribution.png
```

---

## Dataset Schema

| Column | Description |
|---------|--------------|
| tx_id | Transaction ID |
| user_id | Unique user identifier |
| date | Transaction date |
| region | User region |
| merchant | Merchant name or type |
| amount | Transaction amount |

---

## SQL Feature Engineering

Feature generation is handled by `src/queries.sql`.  
It builds temporary SQL views to calculate user statistics and daily activity.
Reformatted here for readability (implicit column aliases in the file, e.g.
`COUNT(*) tx_count`, are spelled out with `AS` below); the logic is identical:

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

---

## Machine Learning

The unsupervised model uses **Isolation Forest** from scikit-learn.

- Detects outliers based on feature deviation  
- Flags top anomalies (typically 2–3% of all transactions)
- Produces a normalized `anomaly_score` between 0 and 1

---

## Visualization

### Anomaly Score Distribution
<img width="1200" height="750" alt="fraud_distribution" src="https://github.com/user-attachments/assets/8adcb416-c0b9-4e3c-8d60-34f11983b3eb" />

This histogram visualizes the distribution of anomaly scores across transactions.  
The right-side tail represents potentially fraudulent or irregular activities.

---

## Tools & Libraries

| Tool | Purpose |
|------|----------|
| **SQLite** | Feature engineering and querying |
| **Python** | Analysis and ML modeling |
| **pandas** | Data manipulation |
| **scikit-learn** | Isolation Forest implementation |
| **matplotlib** | Visualization |

---

## Usage

### Load Data into SQLite
```bash
python src/create_db.py --csv data/transactions.csv --db fraud.db
```

### Run Detection
```bash
python src/detect_fraud_unsupervised.py --db fraud.db --sql src/queries.sql --outdir outputs
```

Optional flags let you tune the Isolation Forest without editing code:
`--contamination` (expected anomaly proportion, default `0.02`) and
`--random-state` (seed, default `7`).

---

## Outputs

| File | Description |
|------|--------------|
| `fraud_scores.csv` | Ranked transactions with anomaly scores |
| `fraud_summary.csv` | User-level fraud summary |
| `fraud_distribution.png` | Histogram of anomaly scores |

---

## Development

Install dev dependencies (linting, formatting, type checking, tests):

```bash
pip install -r requirements-dev.txt
```

`src/` is also an installable package (useful if you want `import src...` to
work from outside the repo, e.g. in a notebook):

```bash
pip install -e .
```

For an exact reproducible environment (pinned transitive dependencies too),
use `requirements-lock.txt` instead of `requirements.txt`.

Run the full quality gate locally, the same checks CI runs on every push/PR
(CI runs this on both Python 3.11 and 3.12):

```bash
make gate       # lint + format-check + typecheck + test
```

or run any step on its own (`make lint`, `make format-check`, `make typecheck`,
`make test`) — see the `Makefile` for the exact commands, which are also what
`.github/workflows/ci.yml` runs.

`outputs/` is generated, not committed — run the two commands in
[Usage](#usage) to produce `fraud_scores.csv`, `fraud_summary.csv`, and
`charts/fraud_distribution.png` locally.

---

## Conclusion

This project showcases a complete unsupervised anomaly detection pipeline.  
It demonstrates how **SQL + Python** can work together to identify rare or irregular financial behaviors, a foundation for fraud prevention and risk analysis systems.
