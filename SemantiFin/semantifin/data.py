from __future__ import annotations
import re
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from .schemas import FinancialDocument
from .utils import clean_text, stable_id

COMPANY_ALIASES = {
    "apple": ("Apple Inc.", "AAPL"), "microsoft": ("Microsoft Corporation", "MSFT"),
    "tesla": ("Tesla, Inc.", "TSLA"), "amazon": ("Amazon.com, Inc.", "AMZN"),
    "alphabet": ("Alphabet Inc.", "GOOGL"), "google": ("Alphabet Inc.", "GOOGL"),
    "nvidia": ("NVIDIA Corporation", "NVDA"), "meta": ("Meta Platforms, Inc.", "META")
}

def infer_entity(text: str, fallback: str = "Unknown Entity"):
    low = text.lower()
    for alias, value in COMPANY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", low): return value
    tick = re.search(r"\b[A-Z]{2,5}\b", text)
    return (fallback, tick.group(0) if tick else None)

def chunk_text(text: str, chunk_words: int = 180, overlap_words: int = 30):
    words = clean_text(text).split()
    if len(words) <= chunk_words: return [" ".join(words)]
    out=[]; step=max(1, chunk_words-overlap_words)
    for i in range(0, len(words), step):
        chunk=words[i:i+chunk_words]
        if chunk: out.append(" ".join(chunk))
        if i+chunk_words >= len(words): break
    return out

def preprocess_documents(docs: list[FinancialDocument], chunk_words=180, overlap_words=30) -> pd.DataFrame:
    rows=[]
    for doc in docs:
        entity,ticker = infer_entity(doc.text, doc.entity_name or "Unknown Entity")
        for j,chunk in enumerate(chunk_text(doc.text,chunk_words,overlap_words)):
            rows.append({
                "document_id":doc.document_id,"chunk_id":f"{doc.document_id}_c{j:03d}","text":chunk,
                "source_dataset":doc.source_dataset,"source_type":doc.source_type,
                "publication_time":doc.publication_time,"canonical_time":doc.canonical_time or doc.event_time or doc.publication_time,
                "entity_name":doc.entity_name or entity,"ticker":doc.ticker or ticker
            })
    return pd.DataFrame(rows)

def demo_documents() -> list[FinancialDocument]:
    base=datetime(2024,1,5)
    samples=[
      ("Apple reported stronger-than-expected quarterly revenue and raised guidance for the next quarter.","news",0),
      ("Apple warned that supply constraints may pressure product availability and margins in the short term.","microblog",18),
      ("Apple disclosed in its filing that regulatory investigations could materially increase legal costs.","filing",35),
      ("Microsoft cloud revenue grew 24 percent, while management maintained a positive full-year outlook.","news",2),
      ("Microsoft noted rising cybersecurity expenses and uncertainty related to regulatory scrutiny.","filing",32),
      ("Tesla deliveries declined and management cited weak demand and pricing pressure.","news",5),
      ("Tesla announced cost reductions and expects automotive margins to stabilize over the medium term.","filing",37),
      ("NVIDIA reported record data-center revenue driven by strong AI demand.","news",7),
      ("NVIDIA faces export restrictions that may limit sales in selected markets.","filing",42),
    ]
    out=[]
    for i,(text,source,days) in enumerate(samples):
        date=(base+timedelta(days=days)).isoformat()
        ent,tick=infer_entity(text)
        out.append(FinancialDocument(stable_id(i,text),text,"demo",source,date,None,date,ent,tick))
    return out
