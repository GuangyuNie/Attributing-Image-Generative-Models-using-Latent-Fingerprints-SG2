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
        self.steps = args.steps  # Num steps for optimizing
        self.save_dir = args.save_dir
        self.relu = torch.nn.ReLU()
        self.log_size = int(math.log(self.img_size, 2))
        self.baseline = False
        self.fix_sigma = True
        
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

        else:
            raise ValueError("Not Avail GANs.")


    def PCA(self):
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
                #latent_std = sum(((latent_out - latent_mean) ** 2) / self.n_mean_latent) ** 0.5
            elif self.model == 'biggan':
                latent_out = truncated_noise_sample(truncation=self.truncation, batch_size=self.n_mean_latent)
                
                pca.fit(latent_out)  # do pca for the style vector data distribution
                var = pca.explained_variance_  # get variance along each pc axis ranked from high to low
                pc = pca.components_  # get the pc ranked from high var to low var
                latent_mean = latent_out.mean(0)
                #latent_std = sum(((latent_out - latent_mean) ** 2) / self.n_mean_latent) ** 0.5
            else:
                raise ValueError("Not supported GAN model.")

        # Get V and U
        var_64 = torch.tensor(var[self.num_main_pc:self.style_space_dim], dtype=torch.float32, device=self.device)  # [64,]
        var_64 = var_64.view(-1, 1)  # [64, 1]
        var_512 = torch.tensor(var, dtype=torch.float32, device=self.device)  # [512,] #todo double check dimension
        var_512 = var_512.view(-1, 1)  # [512, 1]
        sigma_64 = torch.sqrt(var_64) #[64,1]
        sigma_512 = torch.sqrt(var_512) #[512, 1]
        v_cap = torch.tensor(pc[self.num_main_pc:self.style_space_dim, :], dtype=torch.float32,
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
        # self.key = torch.randint(2, (self.key_len, self.batch_size), device=self.device)  # Get random key
        # self.key = torch.as_tensor(np.random.randint(low=0,high=2, size=(self.key_len, self.batch_size))).to(self.device)
        self.key = torch.ones((self.key_len, self.batch_size), device=self.device)  # Get random key
        latent_out = torch.transpose(torch.matmul(u_cap_t, alpha)+self.latent_mean, 0, 1) #to check cosine similarity between alpha used for generating images and reconstructed alpha in classifier code.
        sk_real = torch.multiply(sigma_64, self.key) #considers only positive part.
        # if self.baseline:
        #     noise_sample = torch.randn(self.batch_size, 512, device=self.device)  # get a bunch of Z
        #     latent_out = self.g_ema.style(noise_sample)  # get style vector from Z
        new_latent = latent_out + self.sd_moved * torch.matmul(torch.transpose(sk_real, 0, 1), v_cap)
        

        if self.model == 'sg2':
            if self.style_mixing:
                print('Style mixing...')
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

    def store_results(self, original_image_w0, original_image_wx,reconstructed_w0,reconstucted_wx, watermark_img, iter):
        store_path_w0 = 'image_before_perturb/'
        store_path_wx = 'perturbed_image/'
        store_path_w0_target = 'target_before_perturb/'
        store_path_wx_target = 'target_perturbed/'
        store_path_watermark = 'watermark_gray/'
        isExist = os.path.exists(self.save_dir + store_path_w0)
        if not isExist:
            os.makedirs(self.save_dir + store_path_w0)

        isExist = os.path.exists(self.save_dir + store_path_wx)
        if not isExist:
            os.makedirs(self.save_dir + store_path_wx)

        isExist = os.path.exists(self.save_dir + store_path_w0_target)
        if not isExist:
            os.makedirs(self.save_dir + store_path_w0_target)

        isExist = os.path.exists(self.save_dir + store_path_wx_target)
        if not isExist:
            os.makedirs(self.save_dir + store_path_wx_target)

        isExist = os.path.exists(self.save_dir + store_path_watermark)
        if not isExist:
            os.makedirs(self.save_dir + store_path_watermark)

        for i in range(self.batch_size):
            img_name = self.save_dir + store_path_w0_target + "target_w0_{}.png".format(self.batch_size*iter + i)
            pil_img = Image.fromarray(original_image_w0[i])
            pil_img.save(img_name)
            img_name = self.save_dir + store_path_wx_target + "target_wx_{}.png".format(self.batch_size*iter + i)
            pil_img = Image.fromarray(original_image_wx[i])
            pil_img.save(img_name)
            img_name = self.save_dir + store_path_w0 + "recon_w0_{}.png".format(self.batch_size*iter + i)
            pil_img = Image.fromarray(reconstructed_w0[i])
            pil_img.save(img_name)
            img_name = self.save_dir + store_path_wx + "recon_wx_{}.png".format(self.batch_size*iter + i)
            pil_img = Image.fromarray(reconstucted_wx[i])
            pil_img.save(img_name)
            img_name = self.save_dir + store_path_watermark + "watermark_{}.png".format(self.batch_size*iter + i)
            pil_img = Image.fromarray(watermark_img[i])
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
        "--sample_size", type=int, default=1, help="Number of sample generated"
    )
    parser.add_argument(
        "--sd", type=int, default=1, help="Standard deviation moved"
    )
    parser.add_argument(
        "--steps", type=int, default=1, help="Number of optimization steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for generating images"
    )
    parser.add_argument(
        "--n", type=int, default=1, help="Number of samples for Latin hypercube sampling method"
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

    args = parser.parse_args()


    #shifts = [0,64]
    #shifts = [128,192]
    #shifts = [256,320]
    #shifts = [384,448]
    # shifts = [0]

    if args.model == 'sg2':
        shifts = [0,64,128,256,320,384,448,511]
    elif args.model == 'biggan':
        shifts = [0,16,32,64,127]
    else:
        raise ValueError("Not avail GAN model")

    for shift in shifts:
    # for i in range(510):
    #     shift = 0+i
        torch.manual_seed(1346)
        fixed_sigma = 1
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
        key = []
        wx = []
        w0 = []
        # Get batched
        sigma_448 = sigma_448.repeat(1, optim.batch_size)
        sigma_64 = sigma_64.repeat(1, optim.batch_size)

        tests = args.sample_size  # Number of image tests
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
        rand_alpha = torch.multiply(sigma_448, torch.randn((optim.num_main_pc, optim.batch_size),
                                                           device=optim.device))
        for iter in range(tests):
            loss = []
            a = []
            k = []
            cosine_total = []
            l2_total = []
            # # randn z
            # noise_sample = torch.randn(args.batch_size, 512, device=optim.device)
            # latent_out = optim.g_ema.style(noise_sample)
            # latent_out = latent_out.detach()
            #
            # target_img, target_w0, target_wx = optim.generate_with_latent(latent_out, u_cap_t, sigma_64, v_cap, noise)

            target_img, target_w0, target_wx = optim.generate_with_alpha(rand_alpha, u_cap_t, sigma_64, v_cap, noise)
            target_img = optim.augmentation(target_img)
            sample = samlping.random(n=args.n)  # Sample init guesses
            sample = torch.tensor(sample, dtype=torch.float32, device=optim.device).detach()
            for alpha in sample:
                lr_decay_rate = 4
                lr_segment = lr_decay_rate - 1
                alpha = alpha.view(-1, 1)
                alpha = 2 * torch.multiply(alpha, sigma_448) - 1 * sigma_448
                alpha.requires_grad = True
                key = optim.key_init_guess()
                key.requires_grad = True
                optimizer = torch.optim.Adam([alpha, key], lr=optim.lr)
                early_terminate = False
                lr = optim.lr
                for i in tqdm(range(optim.steps)):
                    optim.g_ema.zero_grad()
                    optimizer.zero_grad()
                    w0 = torch.matmul(torch.transpose(u_cap, 0, 1), alpha)+latent_mean
                    wx = optim.get_new_latent(v_cap, sigma_64, sigmoid(key), w0)
                    estimated_image = optim.generate_image(wx, noise)
                    loss_1 = optim.get_loss(target_img, estimated_image, loss_func="perceptual")

                    loss_total = loss_1 + 0.1 * optim.penalty_1(alpha, max_alpha, min_alpha)
                    if i > optim.steps / 4 and loss_total > 0.2:
                        break
                    if i > optim.steps / 2 and loss_total > 0.1:
                        break
                    if cos(w0.view(-1), target_w0.view(-1)) > 0.995: # Todo: Delete all early termination methods
                        early_terminate = True

                    # Discrete learning rate decay
                    decay = 0.001
                    lr = optim.lr * math.exp(-decay * (i+1))
                    optimizer.param_groups[0]["lr"] = lr

                    loss_total.backward()
                    optimizer.step()
                    cosine = cos(w0.view(-1), target_w0.view(-1))
                    l2 = torch.dist(w0.view(-1), target_w0.view(-1),p=2)
                    if (i + 1) % 100 == 0:
                        print("\nlearning rate is {:.4f}".format(lr))
                        print("Perceptual loss: {:.6f}".format(loss_total.item()))
                        print('Cosine similarity of w0: {:.4f}'
                              .format(cosine))
                        print('l2 distance of w0: {:.4f}'
                              .format(l2))

                if early_terminate == True:
                    break
                else:
                    loss.append(loss_total.item())
                    a.append(alpha)
                    k.append(key)
                    cosine_total.append(cosine)
                    l2_total.append(l2)
            # If early terminated, pick the last one, else, pick the one with min loss
            if early_terminate == True:
                pass
            else:
                min_item = min(loss)
                index = loss.index(min_item)
                alpha = a[index]
                key = k[index]
                cosine = cosine_total[index]
                l2 = l2_total[index]
            alpha = alpha.detach()
            w0 = torch.matmul(torch.transpose(u_cap, 0, 1), alpha)+latent_mean
            w0 = w0.detach()
            estimated_w0 = optim.generate_image(w0, noise)
            target_w0_img = optim.generate_image(target_w0, noise)
            targe_wx_img = optim.generate_image(target_wx,noise)

            wx = optim.get_new_latent(v_cap, sigma_64, sigmoid(key), w0)
            wx_estimated_image = optim.generate_image(wx, noise)
            target_w0_img = optim.make_image(target_w0_img)
            targe_wx_img = optim.make_image(targe_wx_img)

            watermark_img = 10*np.uint8(np.abs(np.int16(targe_wx_img) - np.int16(target_w0_img)))
            watermark_img = optim.rgb2gray(watermark_img)
            watermark_img = np.uint8(watermark_img)


            w0_reconstructed = optim.make_image(estimated_w0)
            wx_reconstructed = optim.make_image(wx_estimated_image)

            key_retrived = torch.round(sigmoid(key))
            print('cosine similarity of key:{}, \ncosine similarity of style vector: {}'
                  .format(cos(key_retrived, optim.key), cos(wx.view(-1), target_wx.view(-1))))
            optim.store_results(target_w0_img,targe_wx_img,w0_reconstructed, wx_reconstructed, watermark_img, iter)
            acc = optim.calculate_classification_acc(torch.round(sigmoid(key)), optim.key)
            print(acc)
            acc_total.append(acc)
            cosine_list.append(cosine)
            l2_list.append(l2)
            if acc == 1.0:
                success += 1
            classification_acc = success / (iter + 1)
            print('Among {} tests, success rate is: {}'.format(iter + 1, classification_acc))
            end = time.time()
            print('time taken for optimization:', end - start)
            with open(optim.save_dir + 'result.txt', 'w') as filehandle:
                for i, listitem in enumerate(acc_total):
                    filehandle.write('\n sample index: {}, key acc: {}, success rate: {},cosine similarity is {:.4f},'
                                     'L2 distance is {:.4f},average cos {}, average l2 {}'.format(i,listitem.item(),
                                      classification_acc,cosine_list[i],l2_list[i],sum(cosine_list) / len(cosine_list),
                                      sum(l2_list) / len(l2_list)))
