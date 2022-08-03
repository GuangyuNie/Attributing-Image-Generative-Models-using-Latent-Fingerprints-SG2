import math
import torch
import torchvision
from pathlib import Path
import torch.utils.data
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
import scipy.stats.qmc as scipy_stats
import lpips
from torch.nn import functional as F
import time
from tqdm import tqdm

# for iter in range(1):
if 1:
    jacobian = torch.empty(92,65536,512)
    jacobian_sigma = torch.empty(92,512)
    jacobian_pc = torch.empty(92,512)
    for i in range(92):
        jacobian[i] = torch.load('./jacobian/jacobian_gray_{}.pt'.format(i))
    print(jacobian.size())
    latent_pc = torch.load('./jacobian/pc.pt')
    latent_sigma = torch.load('./jacobian/sigma512.pt')
    # for a in range(512):
    #     print(torch.norm(latent_pc[a],2))
    eig_vector_observing = 0
    for i in range(92):
        u,s,v = torch.svd(jacobian[i])
        jacobian_sigma[i] = s
        jacobian_pc[i] = v[:,eig_vector_observing]



    jacobian_sigma_std,jacobian_sigma_mean = torch.std_mean(jacobian_sigma,dim=0)
    jacobian_pc_std,jacobian_pc_mean = torch.std_mean(jacobian_pc,dim=0)

    jacobian_pc_mean = jacobian_pc_mean.detach().cpu().numpy()
    jacobian_pc_std = jacobian_pc_std.detach().cpu().numpy()
    jacobian_sigma_mean = jacobian_sigma_mean.detach().cpu().numpy()
    jacobian_sigma_std = jacobian_sigma_std.detach().cpu().numpy()
    latent_sigma = latent_sigma.detach().cpu().numpy()
    latent_pc = latent_pc[eig_vector_observing].detach().cpu().mul(1).numpy()

    # latent_pc, latent_sigma, jacobian_pc, jacobian_sigma
    x = np.linspace(0, 512, 512)
    plt.plot(x,jacobian_sigma_mean, 'k-')
    plt.fill_between(x,jacobian_sigma_mean-jacobian_sigma_std, jacobian_sigma_mean+jacobian_sigma_std)
    plt.plot(x,latent_sigma, 'r-')
    plt.xlabel('number of eig vec')
    plt.ylabel('sigma value(sqrt of eig val)')
    plt.legend(['Jacobian sigma mean','jacobian sigma std','latent sigma'],fontsize='small')
    plt.title('Eigenvalue for Jacobian and latent space')

    plt.show()

    plt.plot(x,jacobian_pc_mean, 'k-')
    plt.fill_between(x,jacobian_pc_mean-jacobian_pc_std, jacobian_pc_mean+jacobian_pc_std)
    plt.plot(x,latent_pc, 'r-')
    plt.xlabel('index of editing direction vector')
    plt.ylabel('value')
    plt.legend(['Jacobian eig vec mean','jacobian eig vec std','latent eig vec'],fontsize='small')
    plt.title('Eigenvector with smallest eigenvalue for Jacobian and latent space')
    plt.show()

    plt.show()



# X_axis = np.arange(len(vec_mean))
# plt.bar(X_axis - 0.2, vec_mean, 0.4, label='jacobian')
# plt.bar(X_axis + 0.2, pc_latent, 0.4, label='latent')
#
# plt.ylabel("Number of Students")
# plt.title("Number of Students in each group")
# plt.legend()
# plt.show()
