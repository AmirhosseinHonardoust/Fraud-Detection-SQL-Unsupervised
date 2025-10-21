import argparse, sqlite3, pandas as pd
from pathlib import Path
parser=argparse.ArgumentParser(); parser.add_argument('--csv',required=True); parser.add_argument('--db',default='fraud.db'); args=parser.parse_args()
df=pd.read_csv(args.csv)
with sqlite3.connect(args.db) as con:
 con.execute('CREATE TABLE IF NOT EXISTS transactions (tx_id INTEGER,user_id TEXT,date TEXT,region TEXT,merchant TEXT,amount REAL)')
 df.to_sql('transactions',con,if_exists='replace',index=False)
print('Loaded',args.csv,'->',args.db)
