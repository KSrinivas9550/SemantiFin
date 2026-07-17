from __future__ import annotations
import numpy as np, pandas as pd
from .embeddings import cosine

DEFAULT_RELIABILITY={"filing":0.95,"financial_news":0.82,"news":0.80,"analyst":0.72,"microblog":0.45,"social":0.40}
class CredibilityFusion:
    def __init__(self,cfg): self.cfg=cfg
    def score(self,records):
        for i,r in enumerate(records):
            sims=[]; contradictions=[]
            for j,s in enumerate(records):
                if i==j or r.entity_name!=s.entity_name: continue
                sim=cosine(r.embedding,s.embedding)
                if sim>=self.cfg.get("similarity_threshold",0.75):
                    agreement=1.0 if r.polarity==s.polarity else 0.0
                    sims.append(sim*agreement); contradictions.append(sim*(1-agreement))
            corr=float(np.mean(sims)) if sims else 0.0
            contra=float(np.mean(contradictions)) if contradictions else 0.0
            source=DEFAULT_RELIABILITY.get(r.source_type,0.60)
            uncertainty=1-r.calibrated_confidence
            cred=(self.cfg["source_weight"]*source+self.cfg["evidence_weight"]*r.evidence_alignment+self.cfg["corroboration_weight"]*corr-self.cfg.get("contradiction_penalty",0.2)*contra-self.cfg.get("uncertainty_penalty",0.1)*uncertainty)
            r.corroboration=corr; r.contradiction=contra; r.credibility=float(np.clip(cred,0,1))
        return records
    def aggregate_windows(self,records,windows):
        by_id={r.insight_id:r for r in records}; out=[]
        for row in windows.itertuples(index=False):
            recs=[by_id[i] for i in row.insight_ids if i in by_id]
            if not recs: continue
            w=np.array([r.credibility or 0 for r in recs]); w=w/w.sum() if w.sum()>0 else np.ones(len(recs))/len(recs)
            embs=np.stack([r.embedding for r in recs]); fused=(w[:,None]*embs).sum(0)
            out.append({"entity":row.entity,"window_start":row.window_start,"fused_embedding":fused,"mean_credibility":float(np.mean([r.credibility for r in recs])),"source_diversity":len(set(r.source_type for r in recs)),"event_distribution":_event_hist(recs,w),"polarity_distribution":_polarity_hist(recs,w),"risk_vector":row.risk_vector,"drift":row.drift,"count":len(recs)})
        return pd.DataFrame(out)
def _event_hist(recs,w):
    labels=["earnings_result","guidance_revision","demand_change","operational_disruption","regulatory_action","litigation","cybersecurity_incident","liquidity_debt","cost_restructuring","other"]
    arr=np.zeros(len(labels))
    for r,x in zip(recs,w): arr[labels.index(r.event_type if r.event_type in labels else "other")]+=x
    return arr
def _polarity_hist(recs,w):
    labels=["positive","neutral","negative","mixed"]; arr=np.zeros(4)
    for r,x in zip(recs,w): arr[labels.index(r.polarity if r.polarity in labels else "neutral")]+=x
    return arr
