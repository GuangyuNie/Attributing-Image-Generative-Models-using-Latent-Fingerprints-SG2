import os
import math
import torch
from sklearn.decomposition import PCA

from model import Generator
from params import opt


class GetPCA:
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




    def perform_pca(self):
        """Do PCA"""
        pca = PCA()
        print("Performing PCA...")
        with torch.no_grad():
            if self.model == 'sg2':
                noise_sample = torch.randn(self.n_mean_latent, self.style_space_dim, device=self.device)  # get a bunch of Z
                latent_out = self.g_ema.style(noise_sample)  # get style vector from Z
                latent_out = latent_out.detach().cpu().numpy()
                pca.fit(latent_out)  # do pca for the style vector data distribution
                var = pca.explained_variance_  # get variance along each pc axis ranked from high to low
                pc = pca.components_  # get the pc ranked from high var to low var
                latent_mean = latent_out.mean(0)
                # latent_std = sum(((latent_out - latent_mean) ** 2) / self.n_mean_latent) ** 0.5
            elif self.model == 'biggan':
                from pytorch_pretrained_biggan import BigGAN, one_hot_from_names, truncated_noise_sample
                latent_out = truncated_noise_sample(truncation=self.truncation, batch_size=self.n_mean_latent)

                pca.fit(latent_out)  # do pca for the style vector data distribution
                var = pca.explained_variance_  # get variance along each pc axis ranked from high to low
                pc = pca.components_  # get the pc ranked from high var to low var
                latent_mean = latent_out.mean(0)
                # latent_std = sum(((latent_out - latent_mean) ** 2) / self.n_mean_latent) ** 0.5
            else:
                raise ValueError("Not supported GAN model.")

        # Get V and U
        var_64 = torch.tensor(var[self.num_main_pc:self.style_space_dim], dtype=torch.float32, device=self.device)  # [64,]
        var_64 = var_64.view(-1, 1)  # [64, 1]
        var_512 = torch.tensor(var, dtype=torch.float32, device=self.device)  # [512,] #
        var_512 = var_512.view(-1, 1)  # [512, 1]
        sigma_64 = torch.sqrt(var_64)  # [64,1]
        sigma_512 = torch.sqrt(var_512)  # [512, 1]
        v_cap = torch.tensor(pc[self.num_main_pc:self.style_space_dim, :], dtype=torch.float32,
                             device=self.device)  # low var pc [64x512]
        u_cap = torch.tensor(pc[0:self.num_main_pc, :], dtype=torch.float32,
                             device=self.device)  # high var pc [448x512]
        pc = torch.tensor(pc, dtype=torch.float32,
                          device=self.device)  # full pc [512x512]

        latent_mean = torch.tensor(latent_mean, dtype=torch.float32,
                                   device=self.device)  # high var pc [1x512]
        self.latent_mean = latent_mean.view(-1, 1)  # + torch.multiply(sigma_512,torch.randn((optim.style_space_dim,1),
        # device=optim.device))

        print("PCA Done")
        return sigma_64, v_cap, u_cap, pc, sigma_512, self.latent_mean