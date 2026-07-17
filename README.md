
[![DOI](https://zenodo.org/badge/1303592394.svg)](https://doi.org/10.5281/zenodo.21406247)
## GitHub Repository

[SemantiFin Repository](https://github.com/KSrinivas9550/SemantiFin)

# SemantiFin

## A Contextual Financial Insight Extraction Framework Using Large Language Models, Temporal Narrative Aggregation, and Credibility-Aware Fusion

SemantiFin is a modular Python research framework for converting heterogeneous financial text into structured, evidence-grounded, temporally aware, credibility-sensitive financial intelligence representations. The framework combines LLM-based event extraction, temporal narrative aggregation, credibility-aware multi-source fusion, and structured representation learning to support downstream financial analytics.

> **Scope:** The current implementation is entirely classical. It generates quantum-compatible representations for future research but does not perform quantum computation or claim quantum advantage.

---

# Features

- Schema-guided financial event and claim extraction
- Entity, polarity, impact, temporal horizon and risk extraction
- Evidence-grounded financial intelligence
- Narrative drift analysis
- Credibility-aware multi-source fusion
- Structured 256-dimensional representation learning
- PCA reduction to 128 dimensions
- Deterministic offline mode
- Optional OpenAI backend
- YAML-based configuration
- Reproducible research pipeline

---

# Framework Pipeline

```text
Financial Text
      │
      ▼
Data Cleaning & Chunking
      │
      ▼
SemantiQ-Extractor
      │
      ▼
Narrative Drift Aggregator
      │
      ▼
Credibility-Aware Fusion
      │
      ▼
Structured Representation Encoder
      │
      ▼
256-D Representation
      │
      ▼
128-D Representation
```

---

# Repository Structure

```text
SemantiFin/
├── configs/
├── data/
├── scripts/
├── semantifin/
├── tests/
├── .env.example
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── run_demo.py
└── README.md
```

---

# Installation

```bash
git clone https://github.com/KSrinivas9550/SemantiFin
cd SemantiFin

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

# Public Datasets

SemantiFin supports:

- Financial PhraseBank
- FiQA Sentiment Dataset
- SEC EDGAR Filings
- Custom financial documents

Download supported datasets:

```bash
python scripts/download_public_datasets.py
```

---

# Quick Start

Run the complete demonstration:

```bash
python run_demo.py
```

or

```bash
semantifin demo --config configs/default.yaml --output outputs
```

Generate figures:

```bash
semantifin figures --config configs/default.yaml --output outputs
```

---

# Configuration

All experiment parameters are stored in:

```text
configs/default.yaml
```

including:

- Extraction backend
- Embedding dimension
- Narrative settings
- Fusion weights
- Encoder parameters
- Random seed

---

# Main Modules

| Module | Description |
|---------|-------------|
| extractor.py | Financial event and claim extraction |
| narrative.py | Temporal narrative aggregation |
| fusion.py | Credibility-aware fusion |
| encoder.py | Structured representation encoder |
| embeddings.py | Text embedding generation |
| evaluation.py | Evaluation metrics |
| pipeline.py | End-to-end workflow |
| visualize.py | Figure generation |

---

# Generated Outputs

A standard run produces:

- Processed financial text
- Structured insight records
- Narrative windows
- Credibility-weighted signals
- 256-D representations
- 128-D representations
- Angle-compatible vectors
- Amplitude-compatible vectors
- Evaluation metrics
- Publication-ready figures

---

# Evaluation

Supported evaluation includes:

- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Evidence Alignment
- Narrative Drift
- Credibility Calibration
- Brier Score
- Representation Stability
- Structural Consistency

---

# Reproducibility

Default settings:

- Random Seed: 42
- Runs: 5
- Window Size: 30 Days
- Embedding Dimension: 768
- Representation Dimension: 256
- Reduced Dimension: 128

---

# Citation

```bibtex
@article{SemantiFin2026,
  title={SemantiFin: A Contextual Financial Insight Extraction Framework Using Large Language Models, Temporal Narrative Aggregation, and Credibility-Aware Fusion},
  author={K. Srinivas },
  year={2026},
  doi={DOI}
}
```

---

# License

This project is licensed under the **MIT License**.


---

# Contact

For questions, bug reports, or collaboration opportunities, please open a GitHub Issue or contact the corresponding author.

---

© 2026 SemantiFin Research Project
