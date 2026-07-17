from __future__ import annotations
import hashlib, json, os, random, re
from pathlib import Path
import numpy as np
import yaml


def load_config(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f)

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)

def ensure_dir(path): Path(path).mkdir(parents=True, exist_ok=True)

def stable_id(*parts: object) -> str:
    return hashlib.sha256("||".join(map(str, parts)).encode()).hexdigest()[:16]

def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def write_jsonl(path, rows):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]
