from semantifin.utils import load_config
from semantifin.pipeline import SemantiFinPipeline
from semantifin.visualize import generate

cfg=load_config("configs/default.yaml")
print(SemantiFinPipeline(cfg,"outputs").run())
generate("outputs")
