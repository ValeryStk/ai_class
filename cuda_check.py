import torch
print(torch.version.cuda)        # Shows the CUDA version PyTorch was built with
print(torch.cuda.is_available()) # True if CUDA is usable