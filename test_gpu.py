import torch

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available")
    device = torch.device('cuda')
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available, using CPU")
    device = torch.device('cpu')

# Create a simple tensor and move it to the GPU
tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
print(f"Tensor on device: {tensor.device}")

# Perform a simple computation
result = tensor * 2
print(f"Result: {result}")

# Check GPU usage
import subprocess

def check_gpu_usage():
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))

check_gpu_usage()
