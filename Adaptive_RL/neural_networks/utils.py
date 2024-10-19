import torch

def local_optimizer(params, lr=3e-4):
    return torch.optim.AdamW(params, lr=float(lr))

def trainable_variables(model):
    return [p for p in model.parameters() if p.requires_grad]

def tile(x, n):
    return x[None].repeat([n] + [1] * len(x.shape))

def merge_first_two_dims(x):
    return x.view(x.shape[0] * x.shape[1], *x.shape[2:])
