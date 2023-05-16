import torch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if torch.cuda.is_available():
    print("cuda version:", torch.version.cuda)
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        device = torch.device(f"cuda:{i}")
        print(f"Device {i}: {device_name}")
        print(
            f"\tMemory Usage: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB / {torch.cuda.max_memory_allocated(device) / 1024 ** 2:.2f} MB")
else:
    print("No CUDA-enabled devices found.")
