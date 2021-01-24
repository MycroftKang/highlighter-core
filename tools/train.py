import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from highlighter.train import Trainer

trainer = Trainer(non_hls_size=5, use_cache=True)
result = trainer.run()

print(result)
