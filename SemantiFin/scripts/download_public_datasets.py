"""Optional downloader for Financial PhraseBank and FiQA using Hugging Face datasets.
Run after: pip install datasets
"""
from pathlib import Path

def main():
    from datasets import load_dataset
    root=Path("data/raw"); root.mkdir(parents=True,exist_ok=True)
    fp=load_dataset("takala/financial_phrasebank","sentences_allagree")
    fp["train"].to_csv(root/"financial_phrasebank.csv")
    fiqa=load_dataset("TheFinAI/fiqa-sentiment-classification")
    for split,ds in fiqa.items(): ds.to_csv(root/f"fiqa_{split}.csv")
    print("Downloaded Financial PhraseBank and FiQA.")
if __name__=="__main__": main()
