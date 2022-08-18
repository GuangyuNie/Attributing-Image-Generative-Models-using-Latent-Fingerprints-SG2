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
        self.n_mean_latent = 1000000  # num of style vector to sample
        self.img_size = args.img_size  # image size
        self.style_space_dim = 512
        self.key_len = args.key_len
        self.batch_size = args.batch_size
        self.mapping_network_layer = 8
        self.num_main_pc = self.style_space_dim - self.key_len  # 512 - 64, num of high var pc
        self.sd_moved = args.sd  # How many standard deviation to move
        self.lr = 0.2
        self.steps = args.steps  # Num steps for optimizing
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
        self.latent_mean = latent_mean.view(-1, 1)

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
        self.key = torch.randint(2, (self.key_len, self.batch_size), device=self.device)  # Get random key
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
        self.key = torch.randint(2, (self.key_len, self.batch_size), device=self.device)  # Get random key

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

    def get_loss(self, img1, img2, loss_func='percept'):
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

    def store_results(self, image,  iter):
        store_path_w0 = './jacobian/image/'
        isExist = os.path.exists(store_path_w0)
        if not isExist:
            os.makedirs(store_path_w0)


        for i in range(self.batch_size):
            img_name = store_path_w0 + "image_{}.png".format(self.batch_size*iter + i)
            pil_img = Image.fromarray(image[i])
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
        "--sample_size", type=int, default=10, help="Number of sample generated"
    )
    parser.add_argument(
        "--sd", type=int, default=1, help="Standard deviation moved"
    )
    parser.add_argument(
        "--steps", type=int, default=2000, help="Number of optimization steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for generating images"
    )
    parser.add_argument(
        "--n", type=int, default=20, help="Number of samples for Latin hypercube sampling method"
    )
    parser.add_argument(
        "--key_len", type=int, default=64, help="Number of digit for the binary key"
    )

    parser.add_argument(
        "--save_dir", type=str, default='./result_images/', help="Directory for image saving"
    )

    parser.add_argument(
        "--augmentation", type=str, default='None',
        help="Augmentation method: Crop, Noise, Blur, Jpeg, Combination "
    )
    #shifts = [1,64]
    #shifts = [128,192]
    #shifts = [256,320]
    # shifts = [384,448]
    shifts = [448]
    store_path_wx_target = './jacobian/'
    isExist = os.path.exists(store_path_wx_target)
    if not isExist:
        os.makedirs(store_path_wx_target)
    for shift in shifts:
        args = parser.parse_args()
        fixed_sigma = 1
        args.save_dir = args.save_dir + "fixed_sigma_{}/shift_{}/".format(fixed_sigma,shift).replace('.', '')
        start = time.time()  # count times to complete
        optim = watermark_optimization(args)
        noise = optim.get_noise()
        tests = args.sample_size  # Number of image tests
        if 1:
            sigma_64, _, _, pc, sigma_512, latent_mean = optim.PCA()
            torch.save(pc, './jacobian/pc.pt')
            torch.save(sigma_512, './jacobian/sigma512.pt')
            v_cap = torch.tensor(pc[shift:shift+64, :], dtype=torch.float32,
                                 device=optim.device)  # low var pc [64x512]
            u_cap = torch.cat([pc[0:shift, :],pc[shift+64:optim.style_space_dim, :]],dim=0)
            u_cap_t = torch.transpose(u_cap, 0, 1)
            if optim.fix_sigma:
                sigma_64 = fixed_sigma * torch.ones_like(sigma_64)
            sigma_448 = torch.cat([sigma_512[0:shift, :],sigma_512[shift+64:optim.style_space_dim, :]],dim=0)

            # Get the boundary of alpha
            max_alpha = 3 * sigma_512
            min_alpha = -3 * sigma_512
            max_alpha = torch.cat([max_alpha[0:shift, :],max_alpha[shift+64:optim.style_space_dim, :]],dim=0)
            min_alpha = torch.cat([min_alpha[0:shift, :],min_alpha[shift+64:optim.style_space_dim, :]],dim=0)
            number_of_images = args.sample_size
            key = []
            wx = []
            w0 = []
            # Get batched
            sigma_448 = sigma_448.repeat(1, optim.batch_size)
            sigma_64 = sigma_64.repeat(1, optim.batch_size)

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
            samlping = scipy_stats.LatinHypercube(d=optim.num_main_pc, centered=True)
            percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=optim.device.startswith("cuda"),gpu_ids=[optim.device_ids])

        for iter in range(tests):
            loss = []
            a = []
            k = []
            cosine_total = []
            l2_total = []
            # # randn z
            noise_sample = torch.randn(args.batch_size, 512, device=optim.device)
            latent_out = optim.g_ema.style(noise_sample)
            # latent_out = latent_out.detach()
            #
            # target_img, target_w0, target_wx = optim.generate_with_latent(latent_out, u_cap_t, sigma_64, v_cap, noise)
            target_img, _ = optim.g_ema(
                [latent_out], noise=noise, input_is_latent=True)
            target_img = optim.augmentation(target_img)
            target_img_pixel = torch.reshape(target_img,(-1,))
            target_img_pixel_i = target_img_pixel[0]
            target_image_range = target_img_pixel.size(dim=0)
            final_jacobian = torch.empty(target_image_range,optim.style_space_dim)
            for i in tqdm(range(target_image_range)):
                jacobian = torch.autograd.grad(target_img_pixel[i],latent_out,retain_graph=True)[0]
                final_jacobian[i] = jacobian
            print(final_jacobian)


            torch.save(final_jacobian, './jacobian/jacobian_{}.pt'.format(iter))
            target_img = optim.make_image(target_img)
            optim.store_results(target_img, iter)
