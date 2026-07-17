from __future__ import annotations
import hashlib, re
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

class TextEmbedder:
    def __init__(self, dimension=768, backend="hashing"):
        self.dimension=dimension; self.backend=backend
        self.vectorizer=HashingVectorizer(n_features=dimension, alternate_sign=False, norm="l2", ngram_range=(1,2))
        self._model=None
        if backend == "transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self._model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            except Exception:
                self.backend="hashing"
    def encode(self, texts):
        if isinstance(texts,str): texts=[texts]
        if self.backend=="transformers" and self._model is not None:
            arr=np.asarray(self._model.encode(texts,normalize_embeddings=True),dtype=float)
            if arr.shape[1] < self.dimension:
                arr=np.pad(arr,((0,0),(0,self.dimension-arr.shape[1])))
            return arr[:,:self.dimension]
        return self.vectorizer.transform(texts).toarray().astype(float)

def cosine(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/den) if den else 0.0
