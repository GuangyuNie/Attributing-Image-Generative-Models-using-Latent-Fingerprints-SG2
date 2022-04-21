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
import lpips
from torch.nn import functional as F
from attack_methods.attack_initializer import attack_initializer

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
        # Get generator
        g_ema = Generator(self.img_size, self.style_space_dim, self.mapping_network_layer)
        g_ema.load_state_dict(torch.load(self.ckpt)["g_ema"], strict=False)  # load ckpt
        #g_ema.eval()  # set to eval mode
        self.g_ema = g_ema.to(self.device)  # push to device


    def PCA(self):
        """Do PCA"""
        pca = PCA()

        if os.path.isfile('./test_images/key_64/pca.pt'):
            pca_dict = torch.load('./test_images/key_64/pca.pt')
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
        latent_out = torch.transpose(torch.matmul(u_cap_t, alpha), 0, 1) #to check cosine similarity between alpha used for generating images and reconstructed alpha in classifier code.
        sk_real = torch.multiply(sigma_64, self.key) #considers only positive part.
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

    def key_init_guess(self, batch_size=1):
        """init guess for key, all zeros (before entering sigmoid function)"""
        return torch.zeros((self.key_len, batch_size), device=self.device)

    def get_new_latent(self, v, s, k, w0): #todo check this function. I changed to work batchwize
        """
        wx = w0 + c(v^T)sk
        w0: style vector prior to perturbation
        c: number of standard deviation moved
        v: last 64 pcs, [64,512]
        s: Diagonal matrix for last 64 pc's standard deviation
        k: 64 digit binary keys
        """
        sigma_diag = torch.diag(s.view(-1))
        #k = k.view(-1, 1)
        vs = torch.matmul(torch.transpose(v, 0, 1), sigma_diag)
        vsk = self.sd_moved * torch.matmul(vs, k)
        return w0 + torch.transpose(vsk,0,1)

    def get_loss(self, img1, img2, loss_func='mse'):
        """Loss function, default: MSE loss"""
        if loss_func == "mse":
            loss = F.mse_loss(img1, img2)
        elif loss_func == "perceptual":
            loss = percept(img1, img2)
        else:
            raise ValueError("Not supported loss func.")
        return loss

    def penalty_1(self, latent, upper, lower):
        """penalty for alpha that exceed the boundary"""
        penalty1 = torch.sum(self.relu(latent - upper))
        penalty2 = torch.sum(self.relu(lower - latent))

        return penalty1 + penalty2

    def calculate_classification_acc(self, approx_key, target_key):
        """Calculate digit-wise key classification accuracy"""
        key_acc = torch.sum(approx_key == target_key)
        acc = key_acc / self.key_len
        return acc


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
        "--sample_size", type=int, default=3000, help="Number of sample generated"
    )
    parser.add_argument(
        "--sd", type=int, default=6, help="Standard deviation moved"
    )

    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for generating images"
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
    parser.add_argument(
        "--lr", type=float, default='1e-3', help="learning rate"
    )

    parser.add_argument(
        "--steps", type=int, default=10, help="optimization steps"
    )

    parser.add_argument(
        "--know_alpha", default=True, action="store_true", help="Know alpha or not"
    )


    args = parser.parse_args()
    args.save_dir = args.save_dir +"key_{}/".format(args.key_len)

    start = time.time()  # count times to complete
    wm_instance = watermark_optimization(args)

    # Frozen Generator
    wm_instance_frozen = watermark_optimization(args)
    wm_instance_frozen.g_ema.eval()
    wm_instance_frozen.g_ema.requires_grad_(False)

    sigma_64, v_cap, u_cap, _, sigma_512, latent_mean, latent_std = wm_instance.PCA()
    # Get projections of the latent mean(for initial guess)

    u_cap_t = torch.transpose(u_cap, 0, 1)
    ata = torch.inverse(torch.matmul(u_cap, torch.transpose(u_cap, 0, 1)))
    projection_u = torch.matmul(torch.matmul(torch.matmul(u_cap_t, ata), u_cap), latent_mean)
    sigma_448 = sigma_512[0:wm_instance.num_main_pc, :]

    # Get the boundary of alpha
    alpha_bar, _ = torch.lstsq(projection_u,
                               torch.transpose(u_cap, 0, 1))  # solve for init of for alpha = [512x1] tensor
    alpha_bar = alpha_bar[0:wm_instance.num_main_pc, :]  # solution for alpha = [448 x 1] tensor
    max_alpha = alpha_bar + 3 * sigma_448
    min_alpha = alpha_bar - 3 * sigma_448

    noise = wm_instance.get_noise()

    number_of_images = args.sample_size
    key = []
    wx = []
    w0 = []
    # Get batched
    alpha_bar = alpha_bar.repeat(1, wm_instance.batch_size)
    sigma_448 = sigma_448.repeat(1, wm_instance.batch_size)
    sigma_64 = sigma_64.repeat(1, wm_instance.batch_size)

    #Fix mapping network
    wm_instance.g_ema.style.requires_grad_(False)
    optimizer_generator = torch.optim.Adam(wm_instance.g_ema.parameters(), lr=args.lr)

    #define loss
    sigmoid = torch.nn.Sigmoid()
    percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=wm_instance.device.startswith("cuda"),
                                   gpu_ids=[wm_instance.device_ids])
    l1_distance = torch.nn.L1Loss()

    #save PCA
    result_file = {
        "sigma_512": sigma_512,
        "sigma_64": sigma_64[:, 0].view(-1, 1),
        "v_cap": v_cap,
        "u_cap": u_cap,
        "latent_mean": latent_mean,
    }
    torch.save(result_file, wm_instance.save_dir + 'pca.pt')


    attack = attack_initializer('Blur', True)

    for i in range(100):
        rand_alpha = torch.multiply(sigma_448, torch.randn((wm_instance.num_main_pc, wm_instance.batch_size),
                                                           device=wm_instance.device)) + alpha_bar
        target_img, target_w0, target_wx = wm_instance_frozen.generate_with_alpha(rand_alpha, u_cap_t, sigma_64, v_cap, noise)

        w0 = target_w0
        wx = target_wx
        key = wm_instance_frozen.key

        #Key estimation stage
        estimation_key = wm_instance.key_init_guess(args.batch_size)
        estimation_key.requires_grad = True

        if not args.know_alpha:
            raise ValueError("Not yet implemented")
        else:
            w0 = target_w0
            optimizer_vector = torch.optim.Adam([estimation_key], lr = wm_instance.lr)

        for iter in range(200):

            estimation_key.requires_grad = True

            wx = wm_instance.get_new_latent(v_cap, sigma_64[:, 0], sigmoid(estimation_key), w0)
            estimated_image = wm_instance.generate_image(wx, noise)


            #Test to check image quality
            #w0_image = wm_instance.make_image(original_image)
            #estimated_image = wm_instance.make_image(estimated_image)
            #wm_instance.store_results(w0_image, estimated_image, j)

            loss_1 = torch.mean(wm_instance.get_loss(attack(target_img), estimated_image, loss_func="perceptual"))

            if not args.know_alpha:
                loss_vector = loss_1 + 0.1 * wm_instance.penalty_1(alpha, max_alpha, min_alpha)
            else:
                loss_vector = loss_1

            optimizer_vector.zero_grad()
            loss_vector.backward(retain_graph=True)
            optimizer_vector.step()

            #estimation_key.requires_grad = False
            #estimation_key.detach()
            #wm_instance.g_ema.train()

            wx = wm_instance.get_new_latent(v_cap, sigma_64[:,0], sigmoid(key-0.1), w0) #Generate Image using True alpha and True beta
            estimated_image = wm_instance.generate_image(wx, noise)
            loss_generator = l1_distance(sigmoid(estimation_key), key) + 0.6 * torch.mean(wm_instance.get_loss(target_img, estimated_image, loss_func="perceptual")) #todo need to check l1 distance
            optimizer_generator.zero_grad()
            loss_generator.backward()
            optimizer_generator.step()



        print("Acc: " + str(torch.sum(torch.abs(torch.round(sigmoid(estimation_key)) - key), dim=0)))
        # Test to check image quality
        w0_image = wm_instance.make_image(target_img)
        estimated_image = wm_instance.make_image(estimated_image)
        wm_instance.store_results(w0_image, estimated_image, i)

    exit()





