from __future__ import annotations
import json, os, re
import numpy as np
import pandas as pd
from .schemas import InsightRecord, EvidenceSpan, NumericClaim
from .utils import stable_id
from .embeddings import TextEmbedder, cosine

EVENT_RULES = {
 "earnings_result":["earnings","quarterly revenue","profit","record revenue"],
 "guidance_revision":["guidance","outlook","expects","forecast"],
 "demand_change":["demand","deliveries","orders"],
 "operational_disruption":["supply constraint","disruption","shortage"],
 "regulatory_action":["regulatory","investigation","scrutiny","restriction"],
 "litigation":["litigation","legal costs","lawsuit"],
 "cybersecurity_incident":["cybersecurity","cyber attack","data breach"],
 "liquidity_debt":["liquidity","debt","cash flow"],
 "cost_restructuring":["cost reduction","restructuring","layoff"],
}
RISK_RULES = {
 "financial_liquidity":["liquidity","debt","cash flow","margin pressure"],
 "operational":["supply","shortage","disruption","availability"],
 "legal_litigation":["legal","litigation","lawsuit","investigation"],
 "regulatory_compliance":["regulatory","compliance","restriction","scrutiny"],
 "macroeconomic":["inflation","interest rate","recession","macro"],
 "geopolitical":["geopolitical","war","sanction","export restriction"],
 "cybersecurity":["cybersecurity","cyber attack","data breach"],
 "esg_reputational":["esg","emission","reputation","sustainability"],
}
POS=["strong","grew","growth","raised","record","positive","improved","stabilize","beat"]
NEG=["declined","weak","pressure","warned","risk","uncertainty","constraint","limit","cost"]

class SemanticExtractor:
    def __init__(self,cfg,embedder:TextEmbedder):
        self.cfg=cfg; self.embedder=embedder
    def _heuristic(self,row):
        text=row.text; low=text.lower()
        event="other"
        for k,terms in EVENT_RULES.items():
            if any(t in low for t in terms): event=k; break
        ps=sum(t in low for t in POS); ns=sum(t in low for t in NEG)
        polarity="positive" if ps>ns else "negative" if ns>ps else "neutral"
        direction={"positive":"bullish","negative":"bearish","neutral":"neutral"}[polarity]
        intensity=min(1.0,0.45+0.12*abs(ps-ns)+0.06*(ps+ns))
        horizon="long" if "long term" in low or "full-year" in low else "medium" if "medium" in low or "next quarter" in low else "short"
        risks=[k for k,terms in RISK_RULES.items() if any(t in low for t in terms)]
        nums=[]
        for m in re.finditer(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>percent|%|million|billion)?",text,re.I):
            val=float(m.group("value")); unit=m.group("unit")
            nums.append(NumericClaim(metric="reported_numeric_value",value=val,unit=unit,normalized_value=val/100 if unit in {"percent","%"} else val))
        evidence=EvidenceSpan(text=text,start_char=0,end_char=len(text))
        claim_vec=self.embedder.encode([text])[0]; ev_vec=self.embedder.encode([evidence.text])[0]
        alignment=cosine(claim_vec,ev_vec)
        schema=1.0; entity_conf=0.9 if row.entity_name!="Unknown Entity" else 0.55
        raw=0.50+0.20*alignment+0.15*schema+0.15*entity_conf
        calibrated=float(np.clip(raw,0,1))
        status="accepted" if calibrated>=self.cfg["confidence_threshold"] and alignment>=self.cfg["evidence_threshold"] else "accepted_low_confidence"
        return InsightRecord(
          stable_id(row.chunk_id,event,text),row.document_id,row.chunk_id,row.entity_name,row.ticker,
          "event" if event!="other" else "claim",event,text,polarity,direction,intensity,horizon,risks,nums,[evidence],
          calibrated,alignment,calibrated,row.source_type,row.canonical_time,status,1,claim_vec.tolist())
    def _openai(self,row):
        try:
            from openai import OpenAI
        except Exception as e: raise RuntimeError("Install openai or use backend=heuristic") from e
        client=OpenAI()
        schema={"entity_name":"string","ticker":"string|null","insight_type":"event|claim|risk|guidance","event_type":"string","claim_text":"string","polarity":"positive|neutral|negative|mixed","impact_direction":"bullish|neutral|bearish|uncertain","impact_intensity":"0..1","temporal_horizon":"short|medium|long|unspecified","risk_categories":"list","evidence_text":"exact substring from input","confidence":"0..1"}
        prompt=f"Extract one evidence-grounded financial insight. Return JSON only matching: {json.dumps(schema)}\nTEXT: {row.text}"
        r=client.chat.completions.create(model=self.cfg.get("model","gpt-4o"),temperature=self.cfg.get("temperature",0.2),top_p=self.cfg.get("top_p",0.9),max_tokens=self.cfg.get("max_output_tokens",1024),messages=[{"role":"system","content":"You are a precise financial information extraction system. Never invent evidence."},{"role":"user","content":prompt}])
        obj=json.loads(r.choices[0].message.content)
        ev=obj.get("evidence_text","")
        start=row.text.find(ev); start=max(0,start); end=start+len(ev)
        emb=self.embedder.encode([obj.get("claim_text",row.text)])[0]
        align=cosine(emb,self.embedder.encode([ev or row.text])[0])
        conf=float(np.clip(0.6*float(obj.get("confidence",0.5))+0.4*align,0,1))
        return InsightRecord(stable_id(row.chunk_id,obj.get("event_type"),obj.get("claim_text")),row.document_id,row.chunk_id,obj.get("entity_name") or row.entity_name,obj.get("ticker") or row.ticker,obj.get("insight_type","claim"),obj.get("event_type","other"),obj.get("claim_text",row.text),obj.get("polarity","neutral"),obj.get("impact_direction","neutral"),float(obj.get("impact_intensity",0.5)),obj.get("temporal_horizon","unspecified"),obj.get("risk_categories",[]),[],[EvidenceSpan(ev or row.text,start,end,align)],float(obj.get("confidence",0.5)),align,conf,row.source_type,row.canonical_time,"accepted" if conf>=self.cfg["confidence_threshold"] and align>=self.cfg["evidence_threshold"] else "accepted_low_confidence",1,emb.tolist())
    def extract(self,chunks:pd.DataFrame):
        records=[]
        for row in chunks.itertuples(index=False):
            records.append(self._openai(row) if self.cfg.get("backend")=="openai" else self._heuristic(row))
        return records
