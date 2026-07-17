from __future__ import annotations
import numpy as np, pandas as pd
from .embeddings import TextEmbedder

RISK_ORDER=["financial_liquidity","operational","legal_litigation","regulatory_compliance","macroeconomic","geopolitical","cybersecurity","esg_reputational"]
class NarrativeDriftAggregator:
    def __init__(self,cfg,embedder:TextEmbedder): self.cfg=cfg; self.embedder=embedder
    def transform(self,records):
        rows=[]
        for r in records:
            rows.append({"entity":r.entity_name,"time":pd.to_datetime(r.canonical_time,errors="coerce"),"confidence":r.calibrated_confidence,"embedding":np.asarray(r.embedding,float),"risks":r.risk_categories,"insight_id":r.insight_id})
        df=pd.DataFrame(rows).dropna(subset=["time"])
        if df.empty: return pd.DataFrame()
        freq=f'{int(self.cfg.get("window_days",30))}D'
        outputs=[]
        for entity,g in df.groupby("entity"):
            g=g.sort_values("time").set_index("time")
            prev_theme=None; prev_smooth=None
            for t,win in g.groupby(pd.Grouper(freq=freq)):
                if win.empty: continue
                conf=np.asarray(win.confidence,float); conf=conf/conf.sum() if conf.sum()>0 else np.ones(len(win))/len(win)
                embs=np.stack(win.embedding.to_list()); theme=(conf[:,None]*embs).sum(0)
                n=np.linalg.norm(theme); theme=theme/n if n else theme
                drift=0.0 if prev_theme is None else float(np.linalg.norm(theme-prev_theme))
                risk=np.zeros(len(RISK_ORDER))
                for w,risks in zip(conf,win.risks):
                    for x in risks:
                        if x in RISK_ORDER: risk[RISK_ORDER.index(x)]+=w
                joint=np.concatenate([theme,risk,[drift]])
                beta=float(self.cfg.get("smoothing_beta",0.3))
                smooth=joint if prev_smooth is None else beta*joint+(1-beta)*prev_smooth
                outputs.append({"entity":entity,"window_start":t.isoformat(),"count":len(win),"drift":drift,"drift_flag":int(drift>=self.cfg.get("drift_threshold",0.25)),"theme_vector":theme,"risk_vector":risk,"joint_vector":joint,"smoothed_vector":smooth,"insight_ids":win.insight_id.tolist()})
                prev_theme=theme; prev_smooth=smooth
        return pd.DataFrame(outputs)
