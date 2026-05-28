import torch
import numpy as np
import random
device = torch.device("cuda:0")

def get_indices(all_points, target_points):
    indices = []
    for target in target_points:
        dist = torch.norm(all_points - target, dim=1)
        idx = torch.argmin(dist)
        indices.append(idx.item())
    return indices

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def gaussian_source(X):
    x0, y0 = 0.0, 0.0
    x, y = X[:, 0:1], X[:, 1:2]
    Q = 100 * torch.exp(-10000 * ((x - x0)**2 + (y - y0)**2))
    return Q
