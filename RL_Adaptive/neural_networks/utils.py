import torch

def local_optimizer(params, lr=3e-4):
    return torch.optim.AdamW(params, lr=lr)

def trainable_variables(model):
    return [p for p in model.parameters() if p.requires_grad]
