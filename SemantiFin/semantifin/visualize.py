from pathlib import Path
import json, pandas as pd, matplotlib.pyplot as plt

def generate(output="outputs"):
    out=Path(output); fig=out/"figures"; fig.mkdir(parents=True,exist_ok=True)
    w=pd.read_csv(out/"narrative_windows.csv")
    if not w.empty:
      for entity,g in w.groupby("entity"):
        plt.figure(figsize=(7,4)); plt.plot(range(len(g)),g["drift"],marker="o"); plt.title(f"Narrative Drift: {entity}",pad=4); plt.xlabel("Window"); plt.ylabel("Drift"); plt.tight_layout(); plt.savefig(fig/f"drift_{entity.replace(' ','_')}.png",dpi=300); plt.close()
    r=[json.loads(x) for x in open(out/"insight_records.jsonl",encoding="utf-8")]
    if r:
      df=pd.DataFrame(r); plt.figure(figsize=(7,4)); df["event_type"].value_counts().plot(kind="bar"); plt.title("Extracted Financial Event Distribution",pad=4); plt.ylabel("Count"); plt.tight_layout(); plt.savefig(fig/"event_distribution.png",dpi=300); plt.close()
