import torch

# Choose the GPU device you want to check
gpu_device = 3
torch.cuda.set_device(gpu_device)

# Print CUDA memory statistics for the selected GPU
print(torch.cuda.memory_summary(abbreviated=False))