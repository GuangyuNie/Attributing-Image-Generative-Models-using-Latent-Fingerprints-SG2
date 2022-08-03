import os
import math

import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from model import Generator
from attack_methods import attack_initializer
import torchvision.transforms as T
import time
import numpy as np
import argparse
import lpips
from torch.nn import functional as F
from attack_methods import rotation

class watermark_optimization:
    def __init__(self, args):
        # Define hyper parameter
        self.device_ids = 0
        self.device = 'cuda:{}'.format(self.device_ids)
        self.ckpt = args.ckpt
        self.n_mean_latent = 10000  # num of style vector to sample
        self.img_size = args.img_size  # image size
        self.style_space_dim = 512
        self.key_len = args.key_len
        self.batch_size = args.batch_size
        self.mapping_network_layer = 8
        self.num_main_pc = self.style_space_dim - self.key_len  # 512 - 64, num of high var pc
        self.sd_moved = args.sd  # How many standard deviation to move
        self.lr = 0.2
        self.save_dir = args.save_dir
        self.relu = torch.nn.ReLU()
        self.log_size = int(math.log(self.img_size, 2))
        self.num_block = self.log_size * 2 - 2
        self.style_mixing = False
        # Get generator
        g_ema = Generator(self.img_size, self.style_space_dim, self.mapping_network_layer)
        g_ema.load_state_dict(torch.load(self.ckpt)["g_ema"], strict=False)  # load ckpt
        g_ema.eval()  # set to eval mode
        self.g_ema = g_ema.to(self.device)  # push to device

    def PCA(self):
        """Do PCA"""
        pca = PCA()
        print("Performing PCA...")
        if os.path.isfile('./PCA/pca.pt'):
            pca_dict = torch.load('./PCA/pca.pt')
            return pca_dict['sigma_64'], pca_dict['v_cap'], pca_dict['u_cap'], None, pca_dict['sigma_512'], pca_dict['latent_mean'], None
        else:
            with torch.no_grad():
                noise_sample = torch.randn(self.n_mean_latent, 512, device=self.device)  # get a bunch of Z
                latent_out = self.g_ema.style(noise_sample)  # get style vector from Z
                latent_out = latent_out.detach().cpu().numpy()
                pca.fit(latent_out)  # do pca for the style vector data distribution
                var = pca.explained_variance_  # get variance along each pc axis ranked from high to low
                pc = pca.components_  # get the pc ranked from high var to low var
                latent_mean = latent_out.mean(0)
                latent_std = sum(((latent_out - latent_mean) ** 2) / self.n_mean_latent) ** 0.5
        # Get V and U
        var_64 = torch.tensor(var[self.num_main_pc:512], dtype=torch.float32, device=self.device)  # [64,]
        var_64 = var_64.view(-1, 1)  # [64, 1]
        var_512 = torch.tensor(var, dtype=torch.float32, device=self.device)  # [64,]
        var_512 = var_512.view(-1, 1)  # [64, 1]
        sigma_64 = torch.sqrt(var_64)
        sigma_512 = torch.sqrt(var_512)
        v_cap = torch.tensor(pc[self.num_main_pc:512, :], dtype=torch.float32,
                             device=self.device)  # low var pc [64x512]
        u_cap = torch.tensor(pc[0:self.num_main_pc, :], dtype=torch.float32,
                             device=self.device)  # high var pc [448x512]
        pc = torch.tensor(pc, dtype=torch.float32,
                          device=self.device)  # full pc [512x512]

        latent_mean = torch.tensor(latent_mean, dtype=torch.float32,
                                   device=self.device)  # high var pc [1x512]
        self.latent_mean = latent_mean.view(-1, 1)
        latent_std = torch.tensor(latent_std, dtype=torch.float32,
                                  device=self.device)  # high var pc [1x512]
        latent_std = latent_std.view(-1, 1)
        print("PCA Done")
        return sigma_64, v_cap, u_cap, pc, sigma_512, self.latent_mean, latent_std

    def generate_with_alpha(self, alpha, u_cap_t, fixed_sigma, pc, noise):
        """
        v_testing:511
        difference:variable, testing:1,2,4,8,16,32,64,128
        for each test run 100 times
        """
        # differences = [0,1,2,4,8,16,32,64,128,256]
        differences = list(range(0, 511))
        loss_total = []
        true_latent = torch.transpose(torch.matmul(u_cap_t, alpha)+self.latent_mean, 0, 1) #to check cosine similarity between alpha used for generating images and reconstructed alpha in classifier code.
        imgs_true, _ = self.g_ema([true_latent], noise=noise, input_is_latent=True)
        for difference in differences:
            new_latent = true_latent + fixed_sigma * pc[511 - difference,:].t()
            imgs_perturb, _ = self.g_ema([new_latent], noise=noise, input_is_latent=True)
            loss = self.get_loss(imgs_true,imgs_perturb)
            loss_total.append(loss.item())
        return np.asarray(loss_total)
        #return loss_total
    def get_noise(self):
        rng = np.random.default_rng(seed=2002)
        log_size = int(math.log(self.img_size, 2))

        noises = [torch.tensor(rng.standard_normal((1, 1, 2 ** 2, 2 ** 2)), dtype=torch.float32, device=self.device)]

        for i in range(3, log_size + 1):
            for _ in range(2):
                noises.append(torch.tensor(np.random.standard_normal((1, 1, 2 ** i, 2 ** i)), dtype=torch.float32,
                                           device=self.device))

        return noises

    def get_loss(self, img1, img2, loss_func='perceptual'):
        """Loss function, default: MSE loss"""
        if loss_func == "mse":
            loss = F.mse_loss(img1, img2)
        elif loss_func == "perceptual":
            loss = percept(img1, img2)
        return loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Image generator for generating perturbed images"
    )
    parser.add_argument(
        "--ckpt", type=str, default='./checkpoint/550000.pt', required=False, help="path to the model checkpoint"
    )
    parser.add_argument(
        "--img_size", type=int, default=256, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--sample_size", type=int, default=100, help="Number of sample generated"
    )
    parser.add_argument(
        "--sd", type=int, default=6, help="Standard deviation moved"
    )

    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for generating images"
    )

    parser.add_argument(
        "--key_len", type=int, default=64, help="Number of digit for the binary key"
    )

    parser.add_argument(
        "--save_dir", type=str, default='./test_images/', help="Directory for image saving"
    )

    parser.add_argument(
        "--augmentation", type=str, default='None', help="Augmentation method: Crop, Noise, Blur, Jpeg, Combination "
    )
    args = parser.parse_args()
    start = time.time()  # count times to complete
    optim = watermark_optimization(args)
    sigma_64, v_cap, u_cap, pc, sigma_512, latent_mean, latent_std = optim.PCA()
    # Get projections of the latent mean(for initial guess)
    v_cap_t = torch.transpose(v_cap, 0, 1)
    ata = torch.inverse(torch.matmul(v_cap, torch.transpose(v_cap, 0, 1)))
    projection_v = torch.matmul(torch.matmul(torch.matmul(v_cap_t, ata), v_cap), latent_mean) #not used
    percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=optim.device.startswith("cuda"),gpu_ids=[optim.device_ids])

    u_cap_t = torch.transpose(u_cap, 0, 1)
    ata = torch.inverse(torch.matmul(u_cap, torch.transpose(u_cap, 0, 1)))
    projection_u = torch.matmul(torch.matmul(torch.matmul(u_cap_t, ata), u_cap), latent_mean)
    sigma_448 = sigma_512[0:optim.num_main_pc, :]

    # Get the boundary of alpha
    alpha_bar, _ = torch.lstsq(projection_u,
                               torch.transpose(u_cap, 0, 1))  # solve for init of for alpha = [512x1] tensor
    alpha_bar = alpha_bar[0:optim.num_main_pc, :]  # solution for alpha = [448 x 1] tensor
    max_alpha = alpha_bar + 3 * sigma_448
    min_alpha = alpha_bar - 3 * sigma_448

    noise = optim.get_noise()

    number_of_images = args.sample_size
    key = []
    wx = []
    w0 = []
    # Get batched
    alpha_bar = alpha_bar.repeat(1, optim.batch_size)
    sigma_448 = sigma_448.repeat(1, optim.batch_size)
    sigma_64 = sigma_64.repeat(1, optim.batch_size)
    fixed_sigma = 1
    num_test = 11
    difference = list(range(0, 511))
    final_loss = np.zeros((num_test,len(difference)))
    for iter in tqdm(range(1)):
        rand_alpha = torch.multiply(sigma_448, torch.randn((optim.num_main_pc, optim.batch_size),
                                                           device=optim.device)) + alpha_bar
        loss = optim.generate_with_alpha(rand_alpha, u_cap_t, fixed_sigma, pc, noise)
        loss = loss.reshape(1,len(difference))
        final_loss[iter] = loss
    final_loss = np.mean(final_loss,0)
    difference = [511 - x for x in difference]
    plt.plot(difference,final_loss)
    plt.xlabel('ith pc')
    plt.ylabel('difference')
    plt.title('perceptual difference vs the ith pc perturbed given fixed sigma=1')
    plt.show()



