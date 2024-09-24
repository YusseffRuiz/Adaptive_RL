import torch

def local_optimizer(params, lr=3e-4):
    return torch.optim.AdamW(params, lr=lr)