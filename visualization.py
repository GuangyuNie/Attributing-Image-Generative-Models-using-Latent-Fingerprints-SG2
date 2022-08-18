"""combine generator and classifier together, classify with raw image"""
import os
import math
import torch
from sklearn.decomposition import PCA
from PIL import Image
from tqdm import tqdm
import lpips
from model import Generator
from torch.nn import functional as F
import scipy.stats.qmc as scipy_stats
import time
import numpy as np
import argparse
from attack_methods import attack_initializer


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
        self.baseline = False
        self.fix_sigma = True
        # Get generator
        g_ema = Generator(self.img_size, self.style_space_dim, self.mapping_network_layer)
        g_ema.load_state_dict(torch.load(self.ckpt)["g_ema"], strict=False)  # load ckpt
        g_ema.eval()  # set to eval mode
        self.g_ema = g_ema.to(self.device)  # push to device

    def PCA(self):
        """Do PCA"""
        pca = PCA()
        print("Performing PCA...")
        with torch.no_grad():
            noise_sample = torch.randn(self.n_mean_latent, 512, device=self.device)  # get a bunch of Z
            latent_out = self.g_ema.style(noise_sample)  # get style vector from Z
            latent_out = latent_out.detach().cpu().numpy()
            pca.fit(latent_out)  # do pca for the style vector data distribution
            var = pca.explained_variance_  # get variance along each pc axis ranked from high to low
            pc = pca.components_  # get the pc ranked from high var to low var
            latent_mean = latent_out.mean(0)
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
        self.latent_mean = latent_mean.view(-1, 1) #+ torch.multiply(sigma_512,torch.randn((optim.style_space_dim,1),
                                                           #device=optim.device))

        print("PCA Done")
        return sigma_64, v_cap, u_cap, pc, sigma_512, self.latent_mean

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
        # self.key = torch.randint(2, (self.key_len, self.batch_size), device=self.device)  # Get random key
        # self.key = torch.as_tensor(np.random.randint(low=0,high=2, size=(self.key_len, self.batch_size))).to(self.device)
        self.key = torch.ones((self.key_len, self.batch_size), device=self.device)  # Get random key
        latent_out = torch.transpose(torch.matmul(u_cap_t, alpha)+self.latent_mean, 0, 1) #to check cosine similarity between alpha used for generating images and reconstructed alpha in classifier code.
        sk_real = torch.multiply(sigma_64, self.key) #considers only positive part.
        # if self.baseline:
        #     noise_sample = torch.randn(self.batch_size, 512, device=self.device)  # get a bunch of Z
        #     latent_out = self.g_ema.style(noise_sample)  # get style vector from Z
        new_latent = latent_out + self.sd_moved * torch.matmul(torch.transpose(sk_real, 0, 1), v_cap)
        if self.style_mixing:
            print('Style mixing...')
            imgs, _ = self.g_ema(
                [latent_out,new_latent], noise=noise, input_is_latent=True,inject_index=self.num_block-1)
        else:
            imgs, _ = self.g_ema(
                [new_latent], noise=noise, input_is_latent=True)

        imgs = imgs.detach()
        latent_out = latent_out.detach()
        new_latent = new_latent.detach()
        return imgs, latent_out, new_latent


    def generate_with_latent(self, latent_out, u_cap_t, sigma_64, v_cap, noise):
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
        # self.key = torch.randint(2, (self.key_len, self.batch_size), device=self.device)  # Get random key
        self.key = torch.ones((self.key_len, self.batch_size), device=self.device)  # Get random key


        ata = torch.inverse(torch.matmul(u_cap, torch.transpose(u_cap, 0, 1)))
        target_w0 = torch.matmul(torch.matmul(torch.matmul(u_cap_t, ata), u_cap), torch.transpose(latent_out,0,1)-latent_mean)

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
        target_w0 = latent_out.detach()
        new_latent = new_latent.detach()
        return imgs, target_w0, new_latent

    def get_loss(self, img1, img2, loss_func='perceptual'):
        """Loss function, default: MSE loss"""
        if loss_func == "mse":
            loss = F.mse_loss(img1, img2)
        elif loss_func == "perceptual":
            loss = percept(img1, img2)
        return loss

    def generate_image(self, style_vector, noise):
        """generate image given style vector and noise"""
        style_vector = style_vector.view(1, -1)
        img_generated, _ = self.g_ema(
            [style_vector], noise=noise, input_is_latent=True)
        return img_generated

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

    def store_results(self, original_image_w0, original_image_wx,watermark_pos,watermark_neg,watermark_pos_gray,watermark_neg_gray,iter):
        watermark_pos_path = 'watermark_pos/'
        watermark_neg_path = 'watermark_neg/'
        watermark_pos_gray_path = 'watermark_pos_gray/'
        watermark_neg_gray_path = 'watermark_neg_gray/'
        store_path_w0_target = 'original/'
        store_path_wx_target = 'watermarked/'
        isExist = os.path.exists(self.save_dir + watermark_pos_path)
        if not isExist:
            os.makedirs(self.save_dir + watermark_pos_path)

        isExist = os.path.exists(self.save_dir + watermark_neg_path)
        if not isExist:
            os.makedirs(self.save_dir + watermark_neg_path)

        isExist = os.path.exists(self.save_dir + watermark_pos_gray_path)
        if not isExist:
            os.makedirs(self.save_dir + watermark_pos_gray_path)

        isExist = os.path.exists(self.save_dir + store_path_wx_target)
        if not isExist:
            os.makedirs(self.save_dir + store_path_wx_target)

        isExist = os.path.exists(self.save_dir + store_path_w0_target)
        if not isExist:
            os.makedirs(self.save_dir + store_path_w0_target)

        isExist = os.path.exists(self.save_dir + watermark_neg_gray_path)
        if not isExist:
            os.makedirs(self.save_dir + watermark_neg_gray_path)



        for i in range(self.batch_size):
            img_name = self.save_dir + store_path_w0_target + "target_w0_{}.png".format(self.batch_size*iter + i)
            pil_img = Image.fromarray(original_image_w0[i])
            pil_img.save(img_name)
            img_name = self.save_dir + store_path_wx_target + "target_wx_{}.png".format(self.batch_size*iter + i)
            pil_img = Image.fromarray(original_image_wx[i])
            pil_img.save(img_name)
            img_name = self.save_dir + watermark_pos_path + "watermark_{}.png".format(self.batch_size*iter + i)
            pil_img = Image.fromarray(watermark_pos[i])
            pil_img.save(img_name)
            img_name = self.save_dir + watermark_neg_path + "watermark_{}.png".format(self.batch_size*iter + i)
            pil_img = Image.fromarray(watermark_neg[i])
            pil_img.save(img_name)

            img_name = self.save_dir + watermark_pos_gray_path + "watermark_{}.png".format(self.batch_size*iter + i)
            pil_img = Image.fromarray(watermark_pos_gray[i])
            pil_img.save(img_name)
            img_name = self.save_dir + watermark_neg_gray_path + "watermark_{}.png".format(self.batch_size*iter + i)
            pil_img = Image.fromarray(watermark_neg_gray[i])
            pil_img.save(img_name)



    def get_noise(self):
        rng = np.random.default_rng(seed=2002)
        log_size = int(math.log(self.img_size, 2))

        noises = [torch.tensor(rng.standard_normal((1, 1, 2 ** 2, 2 ** 2)), dtype=torch.float32, device=self.device)]

        for i in range(3, log_size + 1):
            for _ in range(2):
                noises.append(torch.tensor(np.random.standard_normal((1, 1, 2 ** i, 2 ** i)), dtype=torch.float32,
                                           device=self.device))
        return noises

    def key_init_guess(self):
        """init guess for key, all zeros (before entering sigmoid function)"""
        return torch.zeros((self.key_len, 1), device=self.device)

    def calculate_classification_acc(self, approx_key, target_key):
        """Calculate digit-wise key classification accuracy"""
        key_acc = torch.sum(target_key == approx_key)
        acc = key_acc / self.key_len
        return acc

    def penalty_1(self, latent, upper, lower):
        """penalty for alpha that exceed the boundary"""
        penalty1 = torch.sum(self.relu(latent - upper))
        penalty2 = torch.sum(self.relu(lower - latent))

        return penalty1 + penalty2

    def augmentation(self, target_img):
        """Image augmentation, default is None"""
        if args.augmentation != "None":
            attack = attack_initializer.attack_initializer(args.augmentation,is_train=False)
            target_img = attack(target_img)
        return target_img

    def rgb2gray(self,rgb):
        return np.uint8(np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]))




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
        "--sample_size", type=int, default=1, help="Number of sample generated"
    )
    parser.add_argument(
        "--sd", type=int, default=1, help="Standard deviation moved"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for generating images"
    )
    parser.add_argument(
        "--key_len", type=int, default=1, help="Number of digit for the binary key"
    )
    parser.add_argument(
        "--save_dir", type=str, default='./visualization/', help="Directory for image saving"
    )
    parser.add_argument(
        "--augmentation", type=str, default='None',
        help="Augmentation method: Crop, Noise, Blur, Jpeg, Combination "
    )

    shifts = [0,64,128,256,320,384,448,511]
    fixed_sigma = 3

    # for shift in shifts:
    for i in range(64):
        shift = 448+i
        torch.manual_seed(1346)
        args = parser.parse_args()
        args.save_dir = args.save_dir + "fixed_sigma_{}/shift_{}/".format(fixed_sigma,shift).replace('.', '')
        start = time.time()  # count times to complete
        optim = watermark_optimization(args)
        sigma_64, _, _, pc, sigma_512, latent_mean = optim.PCA()
        v_cap = torch.tensor(pc[shift:shift+optim.key_len, :], dtype=torch.float32,
                             device=optim.device)  # low var pc [64x512]
        u_cap = torch.cat([pc[0:shift, :],pc[shift+optim.key_len:optim.style_space_dim, :]],dim=0)
        u_cap_t = torch.transpose(u_cap, 0, 1)
        if optim.fix_sigma:
            sigma_64 = fixed_sigma * torch.ones_like(sigma_64)
        sigma_448 = torch.cat([sigma_512[0:shift, :],sigma_512[shift+optim.key_len:optim.style_space_dim, :]],dim=0)

        # Get the boundary of alpha
        max_alpha = 3 * sigma_512
        min_alpha = -3 * sigma_512
        max_alpha = torch.cat([max_alpha[0:shift, :],max_alpha[shift+optim.key_len:optim.style_space_dim, :]],dim=0)
        min_alpha = torch.cat([min_alpha[0:shift, :],min_alpha[shift+optim.key_len:optim.style_space_dim, :]],dim=0)
        noise = optim.get_noise()
        number_of_images = args.sample_size
        # Get batched
        sigma_448 = sigma_448.repeat(1, optim.batch_size)
        sigma_64 = sigma_64.repeat(1, optim.batch_size)

        tests = args.sample_size  # Number of image tests
        # Import perceptual loss, cosine similarity and sigmoid function
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        sigmoid = torch.nn.Sigmoid()
        # Import Latin Hypercube Sampling method
        samlping = scipy_stats.LatinHypercube(d=optim.num_main_pc, centered=True)
        percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=optim.device.startswith("cuda"),gpu_ids=[optim.device_ids])
        rand_alpha = torch.multiply(sigma_448, torch.zeros((optim.num_main_pc, optim.batch_size),
                                                           device=optim.device))
        for iter in range(tests):
            loss = []
            a = []
            k = []
            cosine_total = []
            l2_total = []
            target_img, target_w0, target_wx = optim.generate_with_alpha(rand_alpha, u_cap_t, sigma_64, v_cap, noise)
            target_img = optim.augmentation(target_img)
            target_w0_img = optim.generate_image(target_w0, noise)
            target_w0_img = optim.make_image(target_w0_img)
            targe_wx_img = optim.make_image(target_img)

            watermark_pos = np.uint8((np.int16(targe_wx_img) - np.int16(target_w0_img)).clip(0,255))
            watermark_neg = np.uint8((np.int16(target_w0_img) - np.int16(targe_wx_img)).clip(0,255))
            watermark_pos_gray = optim.rgb2gray(watermark_pos)
            watermark_neg_gray = optim.rgb2gray(watermark_neg)

            optim.store_results(target_w0_img,targe_wx_img,watermark_pos, watermark_neg,watermark_pos_gray,watermark_neg_gray,iter)

