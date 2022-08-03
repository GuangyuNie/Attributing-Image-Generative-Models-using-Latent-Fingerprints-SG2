import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
from sklearn.decomposition import PCA
import math
from PIL import Image
import numpy as np
import os

file = torch.load('../000000.pt')
print(file.items)
image_shifted = file['./sample_shifted_500/000000.png']['img']
estimated_latent = file['./sample_shifted_500/000000.png']['latent']
original_latent = np.load('../sample_shifted_500/phi.npy')
original_latent = original_latent[0]
image_shifted = image_shifted.detach().cpu().numpy()
estimated_latent = estimated_latent.detach().cpu().numpy()

print(np.dot(original_latent,estimated_latent))
utils.save_image(
    image_shifted,
    '../test_image.jpg',
    nrow=1,
    normalize=True,
    range=(-1, 1),
)