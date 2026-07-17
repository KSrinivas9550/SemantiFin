from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, brier_score_loss
from scipy.stats import spearmanr, ttest_rel, wilcoxon
from sklearn.metrics import pairwise_distances

def extraction_summary(records):
    return {
      "records":len(records),"schema_validity_rate":float(np.mean([r.validation_status.startswith("accepted") for r in records])) if records else 0,
      "mean_evidence_alignment":float(np.mean([r.evidence_alignment for r in records])) if records else 0,
      "mean_confidence":float(np.mean([r.calibrated_confidence for r in records])) if records else 0,
      "unsupported_claim_rate":float(np.mean([len(r.evidence_spans)==0 for r in records])) if records else 0,
    }
def encoding_metrics(X,Z,fidelity):
    dx=pairwise_distances(X); dz=pairwise_distances(Z)
    tri=np.triu_indices_from(dx,k=1)
    rho=float(spearmanr(dx[tri],dz[tri]).statistic) if len(tri[0])>1 else 1.0
    return {"encoding_fidelity":fidelity,"structural_consistency":rho}
def bootstrap_ci(values,n_boot=1000,seed=42):
    rng=np.random.default_rng(seed); values=np.asarray(values,float)
    means=[rng.choice(values,len(values),replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means,2.5)),float(np.percentile(means,97.5))
def paired_test(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    try: return {"paired_t_p":float(ttest_rel(a,b).pvalue),"wilcoxon_p":float(wilcoxon(a,b).pvalue)}
    except Exception: return {"paired_t_p":None,"wilcoxon_p":None}
