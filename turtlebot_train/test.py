import torch
print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("arch list:", torch.cuda.get_arch_list())
print("device:", torch.cuda.get_device_name(0))
