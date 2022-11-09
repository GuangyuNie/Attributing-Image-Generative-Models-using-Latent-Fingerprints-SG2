import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = "cuda"

def get_rot_mat(theta):
    # theta = torch.tensor(theta)
    rot_matrix = torch.stack([torch.stack(
        [torch.stack([torch.cos(t).unsqueeze(dim=0), -torch.sin(t).unsqueeze(dim=0), torch.zeros(1).to(device)]),
         torch.stack([torch.sin(t).unsqueeze(dim=0), torch.cos(t).unsqueeze(dim=0), torch.zeros(1).to(device)])]) for t in
                         theta]).squeeze()
    return rot_matrix.to(device)


def rot_img(x, theta, dtype):
    # theta = theta.clamp(min=0, max=180)
    rot_mat = get_rot_mat(theta).reshape(-1,2,3).to(device)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype).to(device)
    x = F.grid_sample(x, grid).to(device)
    return x