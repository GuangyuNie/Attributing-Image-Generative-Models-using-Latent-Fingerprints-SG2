import os
import math
import torch
from PIL import Image
from tqdm import tqdm
from model import Generator
from torch.nn import functional as F
import scipy.stats.qmc as scipy_stats
import time
import datetime
import numpy as np
from attack_methods import attack_initializer
from params import opt
from PCA import GetPCA
from utils import *


class GetGen:
    def __init__(self):
        # Define hyper parameter
        self.device_ids = 0
        self.device = 'cuda:{}'.format(self.device_ids)
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
        get_pca = GetPCA()
        self.sigma_64, self.v_cap, self.u_cap, self.pc, self.sigma_512, self.latent_mean = get_pca.perform_pca()


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
            self.g_ema = g_ema.to(self.device)  # push to device
        elif self.model == 'biggan':
            from pytorch_pretrained_biggan import BigGAN, one_hot_from_names, truncated_noise_sample
            model = BigGAN.from_pretrained('biggan-deep-256')
            model.eval()
            self.g_ema = model.to(self.device)
            self.style_space_dim = 128
            self.num_main_pc = self.style_space_dim - self.key_len  # 128 - keylen, num of high var pc.
            self.truncation = 0.4
            self.biggan_label = opt.biggan_label

            self.class_vector = one_hot_from_names([self.biggan_label], batch_size=self.batch_size)
            self.class_vector = torch.from_numpy(self.class_vector).to(self.device)

        else:
            raise ValueError("Not Avail GANs.")

    def generate_with_alpha(self, alpha, u_cap_t, sigma_64, v_cap, noise):
        """
        I = G(wx,n)
        wx = (U^T)*alpha+c(v^T)sk
        v: last 64 pcs, [64,512]
        U: first 448 pcs, [448,512]
        c: number of standard deviation moved
        s: Diagonal matrix for last 64 pc's standard deviation
        k: 64 digit binary keys
        n: fixed noise
        """
        self.key = torch.randint(2, (self.key_len, self.batch_size), device=self.device)  # Get random key
        # self.key = torch.as_tensor(np.random.randint(low=0,high=2, size=(self.key_len, self.batch_size))).to(self.device)
        # self.key = torch.ones((self.key_len, self.batch_size), device=self.device)  # Get random key
        original_latent = torch.transpose(torch.matmul(u_cap_t, alpha) + self.latent_mean, 0,
                                     1)  # to check cosine similarity between alpha used for generating images and reconstructed alpha in classifier code.
        sk_real = torch.multiply(sigma_64, self.key)  # considers only positive part.
        # if self.baseline:
        #     noise_sample = torch.randn(self.batch_size, 512, device=self.device)  # get a bunch of Z
        #     latent_out = self.g_ema.style(noise_sample)  # get style vector from Z
        new_latent = original_latent + self.sd_moved * torch.matmul(torch.transpose(sk_real, 0, 1), v_cap)

        if self.model == 'sg2':
            if self.style_mixing:
                print('Style mixing...')
                imgs, _ = self.g_ema(
                    [original_latent, new_latent], noise=noise, input_is_latent=True, inject_index=self.num_block - 1)
            else:
                imgs, _ = self.g_ema(
                    [new_latent], noise=noise, input_is_latent=True)
        elif self.model == 'biggan':
            imgs = self.g_ema(new_latent, self.class_vector, self.truncation)
        else:
            raise ValueError("Not avail model.")

        imgs = imgs.detach()
        original_latent = original_latent.detach()
        new_latent = new_latent.detach()
        return imgs, original_latent, new_latent, self.key

    def make_dir(self,sigma, shift):
        save_dir = opt.save_dir + "{}/fixed_sigma_{}/shift_{}/".format(opt.augmentation, sigma, shift).replace(
            '.', '')
        return save_dir

    def generate_with_latent(self, u_cap, latent_out, u_cap_t, sigma_64, v_cap, noise):
        """
        I = G(wx,n)
        wx = (U^T)*alpha+c(v^T)sk
        v: last 64 pcs, [64,512]
        U: first 448 pcs, [448,512]
        c: number of standard deviation moved
        s: Diagonal matrix for last 64 pc's standard deviation
        k: 64 digit binary keys
        n: fixed noise
        """
        self.key = torch.randint(2, (self.key_len, self.batch_size), device=self.device)  # Get random key
        #self.key = torch.ones((self.key_len, self.batch_size), device=self.device)  # Get random key


        ata = torch.inverse(torch.matmul(u_cap, torch.transpose(u_cap, 0, 1)))
        target_w0 = torch.matmul(torch.matmul(torch.matmul(u_cap_t, ata), u_cap), torch.transpose(latent_out,0,1)
                                 -self.get_pca.latent_mean)

        sk_real = torch.multiply(sigma_64, self.key) #considers only positive part.
        new_latent = target_w0 + self.sd_moved * torch.matmul(torch.transpose(sk_real, 0, 1), v_cap)
        if self.style_mixing:
            print('Style mixing...')
            imgs, _ = self.g_ema(
                [latent_out,new_latent], noise=noise, input_is_latent=True,inject_index=self.num_block-1)
        else:
            imgs, _ = self.g_ema(
                [new_latent], noise=noise, input_is_latent=True)

        imgs = imgs.detach()
        original_latent = latent_out.detach()
        new_latent = new_latent.detach()
        return imgs, original_latent, new_latent, self.key

    def get_new_latent(self, v, s, k, w0):
        """
        wx = w0 + c(v^T)sk
        w0: style vector prior to perturbation
        c: number of standard deviation moved
        v: last 64 pcs, [64,512]
        s: Diagonal matrix for last 64 pc's standard deviation
        k: 64 digit binary keys
        """
        sigma_diag = torch.diag(s.view(-1))
        k = k.view(-1, 1)
        vs = torch.matmul(torch.transpose(v, 0, 1), sigma_diag)
        vsk = self.sd_moved * torch.matmul(vs, k)
        return w0 + vsk

    def augmentation(self, target_img):
        """Image augmentation, default is None"""
        if opt.augmentation != "None":
            attack = attack_initializer.attack_initializer(opt.augmentation,is_train=False)
            target_img = attack(target_img)
        return target_img

    def generate_image(self, style_vector, noise):
        """generate image given style vector and noise"""
        if self.model == 'sg2':
            style_vector = style_vector.view(1, -1)
            img_generated, _ = self.g_ema([style_vector], noise=noise, input_is_latent=True)
        elif self.model == 'biggan':
            if style_vector.shape != torch.Size([self.batch_size, self.style_space_dim]):
                style_vector = style_vector.t()

            img_generated = self.g_ema(style_vector, self.class_vector, self.truncation)
        else:
            raise ValueError("Not avail GANs")

        return img_generated

    def get_watermarked_image(self):
        rand_alpha = torch.multiply(sigma_448, torch.randn((generator.num_main_pc, opt.batch_size),
                                                           device=opt.device))
        target_img, target_w0, target_wx, key = generator.generate_with_alpha(rand_alpha, u_cap_t, sigma_64, v_cap, noise)
        watermarked_image = make_image(target_img)

        perturbed_image = generator.augmentation(target_img)
        perturbed_image = make_image(perturbed_image)

        original_img = self.generate_image(target_w0,noise)
        original_img = make_image(original_img)


        return original_img, watermarked_image, perturbed_image


if __name__ == "__main__":

    get_pca = GetPCA()
    generator = GetGen()
    shift = opt.shift
    # for i in range(4):
    #     shift = 0+i
    #     torch.manual_seed(1346)
    fixed_sigma = opt.sigma
    save_dir = save_config(generator.make_dir(fixed_sigma,shift))
    start = time.time()  # count times to complete
    v_cap = torch.tensor(generator.pc[shift:shift + opt.key_len, :], dtype=torch.float32,
                         device=opt.device)  # low var pc [64x512]
    u_cap = torch.cat([generator.pc[0:shift, :], generator.pc[shift + opt.key_len:generator.style_space_dim, :]], dim=0)
    u_cap_t = torch.transpose(u_cap, 0, 1)
    sigma_64 = fixed_sigma * torch.ones_like(generator.sigma_64)
    sigma_448 = torch.cat([generator.sigma_512[0:shift, :], generator.sigma_512[shift + opt.key_len:generator.style_space_dim, :]], dim=0)

    # Get the boundary of alpha
    max_alpha = 3 * generator.sigma_512
    min_alpha = -3 * generator.sigma_512
    max_alpha = torch.cat([max_alpha[0:shift, :], max_alpha[shift + opt.key_len:generator.style_space_dim, :]], dim=0)
    min_alpha = torch.cat([min_alpha[0:shift, :], min_alpha[shift + opt.key_len:generator.style_space_dim, :]], dim=0)
    noise = get_noise()
    number_of_images = opt.sample_size
    key = []
    wx = []
    w0 = []
    # Get batched
    sigma_448 = sigma_448.repeat(1, opt.batch_size)
    sigma_64 = sigma_64.repeat(1, opt.batch_size)

    tests = opt.sample_size  # Number of image tests
    early_termination = 0.0005  # Terminate the optimization if loss is below this number
    success = 0  # count number of success
    classification_acc = 0
    acc_total = []
    cosine_list = []
    l2_list = []
    # Import perceptual loss, cosine similarity and sigmoid function
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    sigmoid = torch.nn.Sigmoid()
    # Import Latin Hypercube Sampling method
    samlping = scipy_stats.LatinHypercube(d=generator.num_main_pc, centered=True)
    acc = 0
    for iter in range(tests):
        original_image, watermarked_image, perturbed_image = generator.get_watermarked_image()

        store_results(save_dir,iter,original_image_w0=original_image,original_image_wx=watermarked_image)
