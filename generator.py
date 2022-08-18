import os
import math
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
from attack_methods import rotation
import matplotlib.pyplot as plt
import cv2

from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)

class watermark_optimization:
    def __init__(self, args):
        # Define hyper parameter
        self.device_ids = 0
        self.device = 'cuda:{}'.format(self.device_ids)
        self.n_mean_latent = 10000  # num of style vector to sample
        self.img_size = args.img_size  # image size
        self.key_len = args.key_len
        self.batch_size = args.batch_size
        
        self.sd_moved = args.sd  # How many standard deviation to move
        self.lr = 0.2
        self.save_dir = args.save_dir
        self.relu = torch.nn.ReLU()
        self.log_size = int(math.log(self.img_size, 2))
        self.baseline = False

        self.model = args.model

        if self.model == 'sg2':
            self.ckpt = args.ckpt
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
            
            model = BigGAN.from_pretrained('biggan-deep-256')
            model.eval()
            self.g_ema = model.to(self.device)
            self.style_space_dim = 128
            self.num_main_pc = self.style_space_dim - self.key_len  # 128 - keylen, num of high var pc.
            self.truncation = 0.4
            self.biggan_label = args.biggan_label

            self.class_vector = one_hot_from_names([self.biggan_label], batch_size=self.batch_size)
            self.class_vector = torch.from_numpy(self.class_vector).to(self.device)
            '''
            BigGAN Image generation Code
            # Prepare a input
            truncation = 0.4
            class_vector = one_hot_from_names(['soap bubble', 'coffee', 'mushroom'], batch_size=3)
            noise_vector = truncated_noise_sample(truncation=truncation, batch_size=3)

            # All in tensors
            noise_vector = torch.from_numpy(noise_vector)
            class_vector = torch.from_numpy(class_vector)

            # If you have a GPU, put everything on cuda
            noise_vector = noise_vector.to('cuda')
            class_vector = class_vector.to('cuda')
            model.to('cuda')

            # Generate an image
            with torch.no_grad():
                output = model(noise_vector, class_vector, truncation)

            # If you have a GPU put back on CPU
            output = output.to('cpu')

            # If you have a sixtel compatible terminal you can display the images in the terminal
            # (see https://github.com/saitoha/libsixel for details)
            #display_in_terminal(output)

            # Save results as png images
            save_as_images(output)
            '''

    def PCA(self):
        """Do PCA"""
        pca = PCA()
        print("Performing PCA...")
        if os.path.isfile('./PCA/pca.pt'):
            pca_dict = torch.load('./PCA/pca.pt')
            return pca_dict['sigma_64'], pca_dict['v_cap'], pca_dict['u_cap'], None, pca_dict['sigma_512'], pca_dict['latent_mean'], None
        else:
            with torch.no_grad():
                if self.model == 'sg2':
                    noise_sample = torch.randn(self.n_mean_latent, self.style_space_dim, device=self.device)  # get a bunch of Z
                    latent_out = self.g_ema.style(noise_sample)  # get style vector from Z
                    latent_out = latent_out.detach().cpu().numpy()
                    pca.fit(latent_out)  # do pca for the style vector data distribution
                    var = pca.explained_variance_  # get variance along each pc axis ranked from high to low
                    pc = pca.components_  # get the pc ranked from high var to low var
                    latent_mean = latent_out.mean(0)
                    latent_std = sum(((latent_out - latent_mean) ** 2) / self.n_mean_latent) ** 0.5
                elif self.model == 'biggan':
                    latent_out = truncated_noise_sample(truncation=self.truncation, batch_size=self.n_mean_latent)
                    
                    pca.fit(latent_out)  # do pca for the style vector data distribution
                    var = pca.explained_variance_  # get variance along each pc axis ranked from high to low
                    pc = pca.components_  # get the pc ranked from high var to low var
                    latent_mean = latent_out.mean(0)
                    latent_std = sum(((latent_out - latent_mean) ** 2) / self.n_mean_latent) ** 0.5
                else:
                    raise ValueError("Not supported GAN model.")
        
        
        # Get V and U
        var_64 = torch.tensor(var[self.num_main_pc:self.style_space_dim], dtype=torch.float32, device=self.device)  # [64,]
        var_64 = var_64.view(-1, 1)  # [64, 1]
        var_512 = torch.tensor(var, dtype=torch.float32, device=self.device)  # [512,]
        var_512 = var_512.view(-1, 1)  # [512, 1]
        sigma_64 = torch.sqrt(var_64)
        sigma_512 = torch.sqrt(var_512)
        # if 1:
        #     plt.plot(sigma_512.detach().cpu().numpy())
        #     plt.xlabel('PC index')
        #     plt.ylabel('sigma')
        #     plt.grid()
        #     plt.show()
        #     plt.imshow()
        v_cap = torch.tensor(pc[self.num_main_pc:self.style_space_dim, :], dtype=torch.float32,
                             device=self.device)  # low var pc [64 x style_space_dim]
        u_cap = torch.tensor(pc[0:self.num_main_pc, :], dtype=torch.float32,
                             device=self.device)  # high var pc [448 x style_space_dim]
        pc = torch.tensor(pc, dtype=torch.float32,
                          device=self.device)  # full pc [style_space_dim x style_space_dim]

        latent_mean = torch.tensor(latent_mean, dtype=torch.float32,
                                   device=self.device)  # high var pc [512]
        self.latent_mean = latent_mean.view(-1, 1) #[512x1]
        latent_std = torch.tensor(latent_std, dtype=torch.float32,
                                  device=self.device)  # high var pc [512x1]
        latent_std = latent_std.view(-1, 1)
        print("PCA Done")
        return sigma_64, v_cap, u_cap, pc, sigma_512, self.latent_mean, latent_std

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
        self.key = torch.as_tensor(np.random.randint(low=0,high=2, size=(self.key_len, self.batch_size))).to(self.device)  # Get random key
        latent_out = torch.transpose(torch.matmul(u_cap_t, alpha)+self.latent_mean, 0, 1) #to check cosine similarity between alpha used for generating images and reconstructed alpha in classifier code.
        sk_real = torch.multiply(sigma_64, self.key) #considers only positive part.
        # if self.base_line:
        #     noise_sample = torch.randn(self.batch_size, 512, device=self.device)  # get a bunch of Z
        #     latent_out = self.g_ema.style(noise_sample)  # get style vector from Z
        new_latent = latent_out + self.sd_moved * torch.matmul(torch.transpose(sk_real, 0, 1), v_cap)
        
        if self.model == 'sg2':
            if self.style_mixing:
                imgs, _ = self.g_ema(
                    [latent_out,new_latent], noise=noise, input_is_latent=True,inject_index=self.num_block-1)
            else:
                imgs, _ = self.g_ema(
                    [new_latent], noise=noise, input_is_latent=True)
        elif self.model == 'biggan':
            imgs = self.g_ema(new_latent, self.class_vector, self.truncation)
        else:
            raise ValueError("Not avail model.")

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

    def store_results(self, original_image_w0, original_image_wx,wx_before_augmentation, watermarked,iter):
        store_path_w0 = 'image_before_perturb/'
        store_path_wx = 'perturbed_image/'
        store_path_wa = 'image_before_attack/'
        store_path_watermark = 'watermark/'

        isExist = os.path.exists(self.save_dir + store_path_w0)
        if not isExist:
            os.makedirs(self.save_dir + store_path_w0)

        isExist = os.path.exists(self.save_dir + store_path_wx)
        if not isExist:
            os.makedirs(self.save_dir + store_path_wx)

        isExist = os.path.exists(self.save_dir + store_path_watermark)
        if not isExist:
            os.makedirs(self.save_dir + store_path_watermark)

        if args.augmentation != 'None':
            isExist = os.path.exists(self.save_dir + store_path_wa)
            if not isExist:
                os.makedirs(self.save_dir + store_path_wa)

        for i in range(self.batch_size):
            img_name = self.save_dir + store_path_w0 + "target_w0_{}.png".format(self.batch_size*iter + i)
            pil_img = Image.fromarray(original_image_w0[i])
            pil_img.save(img_name)
            img_name = self.save_dir + store_path_wx + "target_wx_{}.png".format(self.batch_size*iter + i)
            pil_img = Image.fromarray(original_image_wx[i])
            pil_img.save(img_name)
            img_name = self.save_dir + store_path_watermark + "watermark_{}.png".format(self.batch_size*iter + i)
            pil_img = Image.fromarray(watermarked[i])
            pil_img.save(img_name)


            if args.augmentation != 'None':
                img_name = self.save_dir + store_path_wa + "target_wa_{}.png".format(self.batch_size*iter + i)
                pil_img = Image.fromarray(wx_before_augmentation[i])
                pil_img.save(img_name)

    def generate_image(self, style_vector, noise):
        """generate image given style vector and noise"""

        if self.model == 'sg2':
            style_vector = style_vector.view(self.batch_size, -1)
            img_generated, _ = self.g_ema(
                [style_vector], noise=noise, input_is_latent=True)
        elif self.model == 'biggan':
            img_generated = self.g_ema(style_vector, self.class_vector, self.truncation)
        else:
            raise ValueError("Not avail GANs")

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

    def augmentation(self, target_img):
        """Image augmentation, default is None"""
        if args.augmentation == 'Rotation':
            target_img = rotation.rot_img(target_img, 30*torch.pi/180*torch.ones(self.batch_size).to(self.device), dtype=torch.float32)
        elif args.augmentation != "None":
            attack = attack_initializer.attack_initializer(args.augmentation,is_train=False)
            target_img = attack(target_img)
        return target_img

    def rgb2gray(self,rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image generator for generating perturbed images"
    )
    # Added for Biggan tansfer
    parser.add_argument(
        "--model", type=str, default='sg2', required=True, help="GAN model: sg2 | biggan"
    )

    parser.add_argument(
        "--biggan_label", type=str, default='golden retriever', required=False, help="Biggan label to generate image"
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
        "--sd", type=int, default=4, help="Standard deviation moved"
    )

    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for generating images"
    )

    parser.add_argument(
        "--key_len", type=int, default=1, help="Number of digit for the binary key"
    )

    parser.add_argument(
        "--save_dir", type=str, default='./test_images/', help="Directory for image saving"
    )

    parser.add_argument(
        "--augmentation", type=str, default='None', help="Augmentation method: Crop, Noise, Blur, Jpeg, Combination "
    )


    seed = 2000
    for i in range(100):
        args = parser.parse_args()
        number_of_images = args.sample_size
        args.save_dir = args.save_dir + "seed_{}/".format(seed)
        start = time.time()  # count times to complete
        optim = watermark_optimization(args)
        sigma_64, v_cap, u_cap, _, sigma_512, latent_mean, latent_std = optim.PCA()
        # Get projections of the latent mean(for initial guess)
        v_cap_t = torch.transpose(v_cap, 0, 1)
        ata = torch.inverse(torch.matmul(v_cap, torch.transpose(v_cap, 0, 1)))
        projection_v = torch.matmul(torch.matmul(torch.matmul(v_cap_t, ata), v_cap), latent_mean) #not used

        u_cap_t = torch.transpose(u_cap, 0, 1)
        ata = torch.inverse(torch.matmul(u_cap, torch.transpose(u_cap, 0, 1)))
        projection_u = torch.matmul(torch.matmul(torch.matmul(u_cap_t, ata), u_cap), latent_mean)
        sigma_448 = sigma_512[0:optim.num_main_pc, :]


        max_alpha = 3 * sigma_448
        min_alpha = -3 * sigma_448

        noise = optim.get_noise()

        key = []
        wx = []
        w0 = []
        # Get batched
        sigma_448 = sigma_448.repeat(1, optim.batch_size)
        sigma_64 = sigma_64.repeat(1, optim.batch_size)

        torch.manual_seed(seed)
        seed+=1
        rand_alpha = torch.multiply(sigma_448, torch.randn((optim.num_main_pc, optim.batch_size),
                                                           device=optim.device))
        for iter in tqdm(range(int(number_of_images / optim.batch_size) + 1)):

            target_img, target_w0, target_wx = optim.generate_with_alpha(rand_alpha, u_cap_t, sigma_64, v_cap, noise)
            wx_before_augmentation = optim.make_image(target_img)
            original_image = optim.generate_image(target_w0, noise) #8/12 Start From here
            target_img = optim.augmentation(target_img)
            w0_image = optim.make_image(original_image)
            wx_image = optim.make_image(target_img)
            watermark_img = np.uint8(np.abs(np.int16(w0_image) - np.int16(wx_image)))

            watermark_show = watermark_img[0] * 10
            watermark_show = optim.rgb2gray(watermark_show)
            plt.imshow(watermark_show, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
            plt.show()
            for i in range(optim.batch_size):
                wx.append(target_wx[i])
                w0.append(target_w0[i])
                key.append(optim.key[:, i])
            optim.store_results(w0_image, wx_image,wx_before_augmentation,watermark_img,iter)

        result_file = {
            "wx": wx,
            "w0": w0,
            "key": key,
        }
        torch.save(result_file, optim.save_dir + 'test_data.pt')

        result_file = {
            "sigma_512": sigma_512,
            "sigma_64": sigma_64[:, 0].view(-1,1),
            "v_cap": v_cap,
            "u_cap": u_cap,
            "latent_mean": latent_mean,
        }
        torch.save(result_file, optim.save_dir + 'pca.pt')

