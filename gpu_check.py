import torch
import torch.nn as nn

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

# 🔹 Simple GPU test: a dummy matrix multiplication
x = torch.rand(5000, 5000, device=device)
y = torch.rand(5000, 5000, device=device)
z = torch.matmul(x, y)

print("Matrix multiplication successful on", device)
