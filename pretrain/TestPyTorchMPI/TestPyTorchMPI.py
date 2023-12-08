import torch
import os
import torch.distributed as dist

dist.init_process_group('nccl')

half_tensor = torch.rand(5, device='cuda', dtype=torch.float16)
print(half_tensor)
dist.all_reduce(half_tensor)
print(half_tensor)
