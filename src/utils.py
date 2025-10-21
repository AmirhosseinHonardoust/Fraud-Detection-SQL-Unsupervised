from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def ensure_outdir(p):
 p=Path(p); p.mkdir(parents=True, exist_ok=True); return p

def save_csv(df,p):
 p=Path(p); p.parent.mkdir(parents=True, exist_ok=True); df.to_csv(p,index=False); return p

def plot_hist(s,title,out):
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,5)); ax.hist(s,bins=40); ax.set_title(title); ax.set_xlabel('Score'); ax.set_ylabel('Freq'); fig.tight_layout(); fig.savefig(out,dpi=150); plt.close(fig)
