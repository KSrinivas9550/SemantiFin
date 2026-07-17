from __future__ import annotations
import argparse, json
from .utils import load_config
from .pipeline import SemantiFinPipeline
from .experiments import run_ablation, run_sensitivity
from .visualize import generate

def main():
    p=argparse.ArgumentParser(description="SemantiFin research pipeline")
    p.add_argument("command",choices=["demo","ablation","sensitivity","figures"])
    p.add_argument("--config",default="configs/default.yaml"); p.add_argument("--output",default="outputs")
    a=p.parse_args(); cfg=load_config(a.config)
    if a.command=="demo": print(json.dumps(SemantiFinPipeline(cfg,a.output).run(),indent=2))
    elif a.command=="ablation": print(run_ablation(cfg,f"{a.output}/ablation").to_string(index=False))
    elif a.command=="sensitivity": print(run_sensitivity(cfg,f"{a.output}/sensitivity").to_string(index=False))
    else: generate(a.output); print(f"Figures saved to {a.output}/figures")
if __name__=="__main__": main()
