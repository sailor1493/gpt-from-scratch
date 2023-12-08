import torch
import torch.nn as nn
from accelerate import Accelerator
from transformers import (
    get_scheduler,
)

print("Cuda Device Numbers: ", torch.cuda.device_count())

if __name__ == "__main__":
    accelerator = Accelerator()
    model = nn.Linear(20, 30)
    model = accelerator.prepare(model)
    
    print("Finished!")