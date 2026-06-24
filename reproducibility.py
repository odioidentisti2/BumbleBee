import torch

RAND_SEED = 42


def set_torch_seed():
    torch.manual_seed(RAND_SEED)
    torch.cuda.manual_seed_all(RAND_SEED)

def torch_generator():  # For reproducibility in DataLoader
    g = torch.Generator()
    g.manual_seed(RAND_SEED)
    return g

def use_deterministic_algorithms(device):
    if device.type == 'cuda':
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    print("(DETERMINISTIC algorithms)")