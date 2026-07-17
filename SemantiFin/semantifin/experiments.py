from __future__ import annotations
import copy, json
from pathlib import Path
import pandas as pd
from .pipeline import SemantiFinPipeline

def run_ablation(cfg,root="outputs/ablation"):
    variants={
      "full":{},
      "uniform_fusion":{"fusion":{"source_weight":1/3,"evidence_weight":1/3,"corroboration_weight":1/3}},
      "no_corroboration":{"fusion":{"source_weight":0.55,"evidence_weight":0.45,"corroboration_weight":0.0}},
      "no_smoothing":{"narrative":{"smoothing_beta":1.0}},
      "low_dimension":{"encoder":{"reduced_dimension":32}},
      "no_contradiction_penalty":{"fusion":{"contradiction_penalty":0.0}},
    }
    rows=[]
    for name,patch in variants.items():
        c=copy.deepcopy(cfg)
        for sec,vals in patch.items(): c[sec].update(vals)
        m=SemantiFinPipeline(c,f"{root}/{name}").run(); rows.append({"variant":name,**m})
    Path(root).mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(f"{root}/ablation_summary.csv",index=False)
    return pd.DataFrame(rows)

def run_sensitivity(cfg,root="outputs/sensitivity"):
    rows=[]
    for dim in [32,64,128]:
      c=copy.deepcopy(cfg); c["encoder"]["reduced_dimension"]=dim
      m=SemantiFinPipeline(c,f"{root}/dim_{dim}").run(); rows.append({"parameter":"reduced_dimension","value":dim,**m})
    for beta in [0.0,0.3,0.5,0.7]:
      c=copy.deepcopy(cfg); c["narrative"]["smoothing_beta"]=beta
      m=SemantiFinPipeline(c,f"{root}/beta_{beta}").run(); rows.append({"parameter":"smoothing_beta","value":beta,**m})
    Path(root).mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(f"{root}/sensitivity_summary.csv",index=False)
    return pd.DataFrame(rows)
