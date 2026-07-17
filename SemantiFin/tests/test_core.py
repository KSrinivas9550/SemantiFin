from semantifin.data import demo_documents, preprocess_documents
from semantifin.embeddings import TextEmbedder
from semantifin.extractor import SemanticExtractor
from semantifin.utils import load_config

def test_preprocess_and_extract():
    cfg=load_config("configs/default.yaml")
    chunks=preprocess_documents(demo_documents())
    assert len(chunks)>=5
    recs=SemanticExtractor(cfg["extractor"],TextEmbedder()).extract(chunks)
    assert len(recs)==len(chunks)
    assert all(r.evidence_spans for r in recs)
