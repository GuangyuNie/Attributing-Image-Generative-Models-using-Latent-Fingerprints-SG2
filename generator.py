import os
import math
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from model import Generator
import torchvision.transforms as T
import time
import numpy as np
import argparse

class watermark_optimization:
    def __init__(self):
        # Define hyper parameter
        self.device = 'cuda:0'
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
        # Get generator
        g_ema = Generator(self.img_size, self.style_space_dim, self.mapping_network_layer)
        g_ema.load_state_dict(torch.load(self.ckpt)["g_ema"], strict=False)  # load ckpt
        g_ema.eval()  # set to eval mode
        self.g_ema = g_ema.to(self.device)  # push to device

    def PCA(self):
        """Do PCA"""
        pca = PCA()
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
        latent_mean = latent_mean.view(-1, 1)
        latent_std = torch.tensor(latent_std, dtype=torch.float32,
                                  device=self.device)  # high var pc [1x512]
        latent_std = latent_std.view(-1, 1)
        return sigma_64, v_cap, u_cap, pc, sigma_512, latent_mean, latent_std

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
        latent_out = torch.transpose(torch.matmul(u_cap_t, alpha), 0, 1)
        sk_real = torch.multiply(sigma_64, self.key)
        new_latent = latent_out + self.sd_moved * torch.matmul(torch.transpose(sk_real, 0, 1), v_cap)
        imgs, _ = self.g_ema(
            [new_latent], noise=noise, input_is_latent=True)

        imgs = imgs.detach()
        latent_out = latent_out.detach()
        new_latent = new_latent.detach()
        return imgs, latent_out, new_latent

    def make_image(self, tensor):
        """Image postprocessing for output"""
        return (
            tensor.detach()
                .clamp_(min=-1, max=1)
                .add(1)
                .div_(2)
                .mul(255)
                .round()
                .type(torch.uint8)
                .permute(0, 2, 3, 1)
                .to("cpu")
                .numpy()
        )

    def store_results(self, original_image_w0, original_image_wx, iter):
        store_path_w0 = 'image_before_perturb/'
        store_path_wx = 'perturbed_image/'
        isExist = os.path.exists(self.save_dir + store_path_w0)
        if not isExist:
            os.makedirs(self.save_dir + store_path_w0)

        isExist = os.path.exists(self.save_dir + store_path_wx)
        if not isExist:
            os.makedirs(self.save_dir + store_path_wx)
        for i in range(self.batch_size):
            img_name = self.save_dir + store_path_w0 + "target_w0_{}.png".format(self.batch_size*iter + i)
            pil_img = Image.fromarray(original_image_w0[i])
            pil_img.save(img_name)
            img_name = self.save_dir + store_path_wx + "target_wx_{}.png".format(self.batch_size*iter + i)
            pil_img = Image.fromarray(original_image_wx[i])
            pil_img.save(img_name)

    def generate_image(self, style_vector, noise):
        """generate image given style vector and noise"""
        style_vector = style_vector.view(self.batch_size, -1)
        img_generated, _ = self.g_ema(
            [style_vector], noise=noise, input_is_latent=True)
        return img_generated

    def get_noise(self):
        rng = np.random.default_rng(seed=2002)
        log_size = int(math.log(self.img_size, 2))

        noises = [torch.tensor(rng.standard_normal((1, 1, 2 ** 2, 2 ** 2)), dtype=torch.float32, device=self.device)]

        for i in range(3, log_size + 1):
            for _ in range(2):
                noises.append(torch.tensor(np.random.standard_normal((1, 1, 2 ** i, 2 ** i)), dtype=torch.float32,
                                           device=self.device))

        return noises

    def augmentation(self, target_img, aug_method='None'):
        """Image augmentation, default is None"""
        if aug_method == 'blur':
            blurrer = T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5))
            target_img = blurrer(target_img)
        if aug_method == 'noise':
            noise = torch.randn(target_img.size(), device=self.device) * 0.02
            target_img = target_img + noise
        if aug_method == 'rotate':
            rotater = T.RandomRotation(degrees=10)
            target_img = rotater(target_img)
        if aug_method == 'None':
            pass
        if aug_method == 'crop':
            crop_size = 250
            crop = T.CenterCrop(crop_size)
            pad = T.Pad(int((self.img_size - crop_size) / 2))
            target_img = crop(target_img)
            target_img = pad(target_img)
        return target_img


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
        "--sample_size", type=int, default=30, help="Number of sample generated"
    )
    parser.add_argument(
        "--sd", type=int, default=6, help="Standard deviation moved"
    )

    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for generating images"
    )

    parser.add_argument(
        "--key_len", type=int, default=64, help="Number of digit for the binary key"
    )

    parser.add_argument(
        "--save_dir", type=str, default='./test_images/', help="Directory for image saving"
    )
    parser.add_argument(
        "--augmentation", type=str, default='None', help="Augmentation method"
    )
    args = parser.parse_args()

    start = time.time()  # count times to complete
    optim = watermark_optimization()
    sigma_64, v_cap, u_cap, _, sigma_512, latent_mean, latent_std = optim.PCA()
    # Get projections of the latent mean(for initial guess)
    v_cap_t = torch.transpose(v_cap, 0, 1)
    ata = torch.inverse(torch.matmul(v_cap, torch.transpose(v_cap, 0, 1)))
    projection_v = torch.matmul(torch.matmul(torch.matmul(v_cap_t, ata), v_cap), latent_mean)

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
    for iter in tqdm(range(int(number_of_images / optim.batch_size) + 1)):
        rand_alpha = torch.multiply(sigma_448, torch.randn((optim.num_main_pc, optim.batch_size),
                                                           device=optim.device)) + alpha_bar
        target_img, target_w0, target_wx = optim.generate_with_alpha(rand_alpha, u_cap_t, sigma_64, v_cap, noise)
        original_image = optim.generate_image(target_w0, noise)
        w0_image = optim.make_image(original_image)
        wx_image = optim.make_image(target_img)
        for i in range(optim.batch_size):
            wx.append(target_wx[i])
            w0.append(target_w0[i])
            key.append(optim.key[:, i])
        optim.store_results(w0_image, wx_image, iter)

    result_file = {
        "wx": wx,
        "w0": w0,
        "key": key,
    }
    torch.save(result_file, optim.save_dir + 'test_data.pt')

    result_file = {
        "sigma_512": sigma_512,
        "sigma_64": sigma_64[:, 0],
        "v_cap": v_cap,
        "u_cap": u_cap,
        "latent_mean": latent_mean,
    }
    torch.save(result_file, optim.save_dir + 'pca.pt')

