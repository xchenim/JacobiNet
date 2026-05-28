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


def compute_normal_vector(points, reference_point):
    p1 = points[0].cpu().detach().numpy()
    p2 = points[-1].cpu().detach().numpy()
    direction_vector = p2 - p1
    normal_vector = np.array([-direction_vector[1], direction_vector[0]])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    reference_point = reference_point.cpu().detach().numpy()
    vector_to_reference = reference_point - p1
    dot_product = np.dot(normal_vector, vector_to_reference)
    if dot_product < 0:
        normal_vector = -normal_vector

    return torch.tensor(normal_vector, dtype=torch.float32).to(device)


def parabolic_velocity(inlet_points, outlet_points):
    device = inlet_points.device
    x = inlet_points[:, 0]
    y = inlet_points[:, 1]

    r2 = x*x + y*y
    mag = torch.clamp(1.0 - r2, min=0.0)

    Ni = inlet_points.shape[0]
    vel = torch.zeros((Ni, 3), dtype=torch.float32, device=device)
    vel[:, 2] = mag


    print(f"Max velocity magnitude: {mag.max():.6f}")
    print(f"Mean velocity magnitude: {mag.mean():.6f}")

    return vel
