import os

import torch


def is_gpu():
    return os.getenv("GPU","no").lower() in ["1",1,"yes","true","t"]

def gpu():
    if is_gpu():
        torch.cuda.set_device(int(os.getenv("CUDA_DEVICE", 0)))
        return True
    return False