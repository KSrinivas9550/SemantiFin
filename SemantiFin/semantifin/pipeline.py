from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from .utils import ensure_dir, set_seed, write_jsonl
from .data import demo_documents, preprocess_documents
from .embeddings import TextEmbedder
from .extractor import SemanticExtractor
from .narrative import NarrativeDriftAggregator
from .fusion import CredibilityFusion
from .encoder import StructuredEncoder
from .evaluation import extraction_summary, encoding_metrics

class SemantiFinPipeline:
    def __init__(self,cfg,output_dir="outputs"):
        self.cfg=cfg; self.output=Path(output_dir); ensure_dir(self.output)
        set_seed(cfg["project"].get("seed",42))
        self.embedder=TextEmbedder(cfg["embedding"].get("dimension",768),cfg["embedding"].get("backend","hashing"))
    def run(self,documents=None):
        documents=documents or demo_documents()
        chunks=preprocess_documents(documents,self.cfg["data"].get("chunk_words",180),self.cfg["data"].get("overlap_words",30))
        chunks.to_csv(self.output/"processed_chunks.csv",index=False)
        extractor=SemanticExtractor(self.cfg["extractor"],self.embedder); records=extractor.extract(chunks)
        fusion=CredibilityFusion(self.cfg["fusion"]); records=fusion.score(records)
        write_jsonl(self.output/"insight_records.jsonl",[r.to_dict() for r in records])
        ncfg={**self.cfg["narrative"],"window_days":self.cfg["data"].get("window_days",30)}
        windows=NarrativeDriftAggregator(ncfg,self.embedder).transform(records)
        _save_vector_df(windows,self.output/"narrative_windows.csv")
        fused=fusion.aggregate_windows(records,windows)
        _save_vector_df(fused,self.output/"fused_signals.csv")
        if fused.empty: raise RuntimeError("No temporal windows produced. Check timestamps.")
        enc=StructuredEncoder(self.cfg["encoder"]); X,Z,A,AMP,fidelity=enc.fit_transform(fused)
        np.save(self.output/"representations_256.npy",X); np.save(self.output/"representations_128.npy",Z)
        np.save(self.output/"angle_parameters.npy",A); np.save(self.output/"amplitude_parameters.npy",AMP)
        enc.save(self.output/"encoder.joblib")
        metrics={**extraction_summary(records),**encoding_metrics(X,Z,fidelity),"windows":len(windows),"mean_drift":float(windows.drift.mean()),"mean_credibility":float(np.mean([r.credibility for r in records]))}
        with open(self.output/"metrics.json","w") as f: json.dump(metrics,f,indent=2)
        return metrics

def _save_vector_df(df,path):
    if df.empty: df.to_csv(path,index=False); return
    out=df.copy()
    for c in out.columns:
        if len(out) and isinstance(out.iloc[0][c],(np.ndarray,list)):
            out[c]=out[c].apply(lambda x: json.dumps(np.asarray(x).tolist()))
    out.to_csv(path,index=False)
