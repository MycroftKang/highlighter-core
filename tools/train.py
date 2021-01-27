import os
import sys

from highlighter.train import Trainer

trainer = Trainer(non_hls_size=5, use_cache=True)
result = trainer.run()

print(result)
