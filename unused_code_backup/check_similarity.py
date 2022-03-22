import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
from sklearn.decomposition import PCA
import math
from PIL import Image
import numpy as np
import os
device = 'cuda:0'
file = torch.load('../projection_Test/normalized_trial_m_3/normalized.pt')
print(file.items)
output_all = []
for i, item in enumerate(file):
    #original_latent = np.load('../projection_Test/true_latent.npy')
    original_latent = np.load('../sample_shifted_500/phi.npy')
    estimated_latent = file[item]['latent']
    original_latent = original_latent[i]
    original_latent = np.reshape(original_latent, 512)
    original_latent = torch.tensor(original_latent, device=device)

    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    output = cos(estimated_latent, original_latent)
    output_all.append(output.detach().cpu().numpy())

    print(output)
print(np.mean(output_all))


def total_variation_loss(img, weight):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)