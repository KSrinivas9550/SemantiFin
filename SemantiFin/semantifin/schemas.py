from __future__ import annotations
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any

@dataclass
class FinancialDocument:
    document_id: str
    text: str
    source_dataset: str = "demo"
    source_type: str = "news"
    publication_time: str | None = None
    event_time: str | None = None
    canonical_time: str | None = None
    entity_name: str | None = None
    ticker: str | None = None
    cik: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass
class EvidenceSpan:
    text: str
    start_char: int
    end_char: int
    semantic_similarity: float = 0.0
    def to_dict(self): return asdict(self)

@dataclass
class NumericClaim:
    metric: str
    value: float | None = None
    unit: str | None = None
    currency: str | None = None
    period: str | None = None
    normalized_value: float | None = None
    def to_dict(self): return asdict(self)

@dataclass
class InsightRecord:
    insight_id: str
    document_id: str
    chunk_id: str
    entity_name: str
    ticker: str | None
    insight_type: str
    event_type: str
    claim_text: str
    polarity: str
    impact_direction: str
    impact_intensity: float
    temporal_horizon: str
    risk_categories: list[str]
    numeric_claims: list[NumericClaim]
    evidence_spans: list[EvidenceSpan]
    llm_confidence: float
    evidence_alignment: float
    calibrated_confidence: float
    source_type: str
    canonical_time: str | None
    validation_status: str = "accepted"
    extraction_attempt: int = 1
    embedding: list[float] | None = None
    credibility: float | None = None
    corroboration: float | None = None
    contradiction: float | None = None
    def to_dict(self):
        d = asdict(self)
        return d
