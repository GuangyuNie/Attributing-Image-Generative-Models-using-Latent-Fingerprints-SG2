import torch
from model import Generator
from scipy.optimize import *
import numpy as np
from multiprocessing import Pool
import os
from PIL import Image
from torchvision import utils
import math

device = 'cuda:0'
batch = 1
latent = 512
ckpt = './checkpoint/550000.pt'
g_ema = Generator(256, latent, n_mlp=8, channel_multiplier=2).to(device)
checkpoint = torch.load(ckpt)
g_ema.load_state_dict(checkpoint["g_ema"], strict=False)
data_sample = np.load('data_sample.npy')
eig_vec = np.load('eig_vec.npy')
eig_val = np.load('eig_val.npy')

def generate(latent, w, g_ema, device, style_mixing):
    torch.manual_seed(2022)
    # training dataset seed 2022
    with torch.no_grad():
        g_ema.eval()
        if style_mixing:
            image_shifted, _ = g_ema(
                [w, latent], input_is_latent=True, truncation=1,
                truncation_latent=None)
        else:
            image_shifted, _ = g_ema(
                [latent], input_is_latent=True, truncation=1,
                truncation_latent=None)
    return image_shifted

def initial_w(data_sample):
    return np.mean(data_sample,axis=0)

def initial_r():
    return np.transpose(np.zeros(64))

def get_pca(eig_vec):
    starting_pos = 512-64
    return eig_vec[starting_pos:512,:]

def get_var(eig_val):
    starting_pos = 512-64
    print(eig_val.shape)
    return eig_val[starting_pos:512]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def l2_norm(x):
    return np.sqrt(np.sum(x ** 2))


def objective_function(x, original_image,pca,eig_value):
    r = x[:64]
    r = r.reshape(1,64)
    w = x[64:]
    w = w.reshape(1,512)
    key = np.multiply(sigmoid(r), 6*np.sqrt(eig_value))
    latent = w+np.dot(key,pca)
    latent = torch.tensor(latent,dtype=torch.float32,device = device)
    w = torch.tensor(w, dtype=torch.float32,device = device)
    generated = generate(latent,w,g_ema,device,style_mixing=True)
    generated = generated.detach().cpu().numpy()
    generated = generated.reshape(256,256,3)
    loss = (l2_norm(original_image - generated)) ** 2
    print(loss)
    return loss

pca = get_pca(eig_vec)
eig_val = get_var(eig_val)
original_image = np.load('./sample_shifted_10/image_0.npy')
original_image = original_image.reshape((256,256,3))
guess = np.concatenate((initial_r(),initial_w(data_sample)))
result = minimize(objective_function, guess, args = (original_image,pca,eig_val), method='Powell', options={'ftol':2000})

if result.success:
    perturb = np.round(sigmoid(result.x[:64]))
    latent = result.x[64:]
    print(perturb)
    print(latent)
    np.save('./optimization_key_0.npy', perturb)
    np.save('./optimization_0.npy', latent)
else:
    raise ValueError(result.message)

