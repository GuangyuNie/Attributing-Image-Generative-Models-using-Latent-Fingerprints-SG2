import os
import math
import torch
from sklearn.decomposition import PCA
from PIL import Image
from tqdm import tqdm
import custom_lpips
from model import Generator
from torch.nn import functional as F
import time
import numpy as np
import argparse
from attack_methods import attack_initializer

from params import opt


class GetClassifier:
    def __init__(self):
        # Define hyper parameter
        self.n_mean_latent = 10000  # num of style vector to sample
        self.img_size = opt.img_size  # image size
        self.key_len = opt.key_len
        self.batch_size = opt.batch_size

        self.sd_moved = opt.sd  # How many standard deviation to move
        self.lr = 0.2
        self.steps = opt.steps  # Num steps for optimizing
        self.save_dir = opt.save_dir
        self.relu = torch.nn.ReLU()
        self.log_size = int(math.log(self.img_size, 2))
        self.baseline = False
        self.fix_sigma = True

        self.model = opt.model
        self.percept = custom_lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=opt.device)

        if self.model == 'sg2':
            self.ckpt = opt.ckpt
            self.style_space_dim = 512
            self.mapping_network_layer = 8
            self.num_block = self.log_size * 2 - 2
            self.style_mixing = False
            self.num_main_pc = self.style_space_dim - self.key_len  # 512 - 64, num of high var pc
            # Get generator
            g_ema = Generator(self.img_size, self.style_space_dim, self.mapping_network_layer)
            g_ema.load_state_dict(torch.load(self.ckpt)["g_ema"], strict=False)  # load ckpt
            g_ema.eval()  # set to eval mode
            self.g_ema = g_ema.to(opt.device)  # push to device
        elif self.model == 'biggan':
            from pytorch_pretrained_biggan import BigGAN, one_hot_from_names, truncated_noise_sample
            model = BigGAN.from_pretrained('biggan-deep-256')
            model.eval()
            self.g_ema = model.to(opt.device)
            self.style_space_dim = 128
            self.num_main_pc = self.style_space_dim - self.key_len  # 128 - keylen, num of high var pc.
            self.truncation = 0.4
            self.biggan_label = opt.biggan_label

            self.class_vector = one_hot_from_names([self.biggan_label], batch_size=self.batch_size)
            self.class_vector = torch.from_numpy(self.class_vector).to(opt.device)

        else:
            raise ValueError("Not Avail GANs.")



