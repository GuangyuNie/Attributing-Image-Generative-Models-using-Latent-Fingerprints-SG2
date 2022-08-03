import os
import math
import torch
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
import numpy as np
import lpips
from model import Generator
import matplotlib.pyplot as plt


class watermark_optimization:
    def __init__(self):
        # Define hyper parameter
        # torch.manual_seed(2009)
        #torch.manual_seed(1994)
        self.device = 'cuda:0'
        self.ckpt = './checkpoint/550000.pt'
        self.n_mean_latent = 10000  # num of style vector to sample
        self.steps = 2000
        self.img_size = 256  # image size
        self.style_space_dim = 512
        self.mapping_network_layer = 8
        self.resize = min(self.img_size, 256)
        self.critical_point = 448  # 512 - 64, num of high var pc
        self.eig_index = 0  # Index shifted for the last 64 digits (0~63)
        self.sd_moved = 12  # How many standard deviation to move
        self.key_len = self.style_space_dim - self.critical_point
        self.key = torch.randint(2, (self.key_len, 1), device=self.device)  # Get random key
        self.save_dir = './projection_Test/trial_3_original_4/'
        self.relu = torch.nn.ReLU()
        self.noise = False
        # Get generator
        g_ema = Generator(self.img_size, self.style_space_dim, self.mapping_network_layer)
        g_ema.load_state_dict(torch.load(self.ckpt)["g_ema"], strict=False)  # load ckpt
        g_ema.eval()  # set to eval mode
        self.g_ema = g_ema.to(self.device)  # push to device
        if 1:
            self.zero_noise = []
            noises = self.g_ema.make_noise()  # get init guess of noise
            for noise in noises:
                noise = torch.zeros_like(noise)
                self.zero_noise.append(noise)

    def PCA(self):
        pca = PCA()
        with torch.no_grad():
            noise_sample = torch.randn(self.n_mean_latent, 512, device=self.device)  # get a bunch of Z
            latent_out = self.g_ema.style(noise_sample)  # get style vector from Z
            latent_out = latent_out.detach().cpu().numpy()
            pca.fit(latent_out)  # do pca for the style vector data distribution
            var = pca.explained_variance_  # get variance along each pc axis ranked from high to low
            pc = pca.components_  # get the pc ranked from high var to low var
            latent_mean = latent_out.mean(0)
            latent_std = sum(((latent_out - latent_mean)**2) / self.n_mean_latent) ** 0.5
        # Get V and U
        var_64 = torch.tensor(var[self.critical_point:512], dtype=torch.float32, device=self.device)  # [64,]
        var_64 = var_64.view(-1, 1)  # [64, 1]
        var_512 = torch.tensor(var, dtype=torch.float32, device=self.device)  # [64,]
        var_512 = var_512.view(-1, 1)  # [64, 1]
        sigma_64 = torch.sqrt(var_64)
        sigma_512 = torch.sqrt(var_512)
        v_cap = torch.tensor(pc[self.critical_point:512, :], dtype=torch.float32,
                             device=self.device)  # low var pc [64x512]
        u_cap = torch.tensor(pc[0:self.critical_point, :], dtype=torch.float32,
                             device=self.device)  # high var pc [448x512]
        pc = torch.tensor(pc, dtype=torch.float32,
                             device=self.device)  # high var pc [448x512]


        latent_mean = torch.tensor(latent_mean, dtype=torch.float32,
                             device=self.device)  # high var pc [1x512]
        latent_mean = latent_mean.view(-1, 1)
        latent_std = torch.tensor(latent_std, dtype=torch.float32,
                             device=self.device)  # high var pc [1x512]
        latent_std = latent_std.view(-1, 1)
        return sigma_64, v_cap, u_cap, pc, sigma_512, latent_mean, latent_std

    def generate_target_image(self, sigma_64, v_cap):
        noise_sample = torch.randn(1, 512, device=self.device)  # get a bunch of Z
        latent_out = self.g_ema.style(noise_sample)  # get style vector W from Z
        sk_real = torch.multiply(sigma_64, self.key)
        new_latent = latent_out + self.sd_moved * torch.matmul(torch.transpose(sk_real, 0, 1), v_cap)
        if self.noise == False:
            imgs, _ = self.g_ema(
                [new_latent], noise = self.zero_noise,input_is_latent=True)
        else:
            imgs, _ = self.g_ema(
                [new_latent], input_is_latent=True)
        latent_out.detach()
        new_latent.detach()
        return imgs, latent_out, new_latent

    def get_loss(self,img1,img2,loss_func='mse'):
        if loss_func == "mse":
            loss = F.mse_loss(img1,img2)
        elif loss_func == "perceptual":
            loss = percept(img1,img2)
        return loss

    def generate_image(self, style_vector):
        style_vector = style_vector.view(1,-1)

        if self.noise == False:
            img_generated, _ = self.g_ema(
                [style_vector], noise = self.zero_noise,input_is_latent=True)
        else:
            img_generated, _ = self.g_ema(
                [style_vector], input_is_latent=True)
        return img_generated

    def get_grad(self, y, x, second_order=False):
        """Return partial y partial x"""
        grad = torch.autograd.grad(y, x, create_graph=True)
        grad = torch.transpose(grad[0], 0, 1)
        if second_order==True:
            grad_2nd = []
            for i in range(grad[0]):
                grad_i = grad_phi[i]
                grad_2nd_i = torch.autograd.grad(grad_i, x, retain_graph=True)
                grad_2nd.append(grad_2nd_i[0])
            grad_2nd = torch.stack(grad_2nd, 1)[0].to(self.device)
            return grad, grad_2nd
        return grad

    def get_new_latent(self, v, s, k, w0):
        sigma_diag = torch.diag(s.view(-1))
        k = k.view(-1,1)
        vs = torch.matmul(torch.transpose(v, 0, 1), sigma_diag)
        vsk = self.sd_moved * torch.matmul(vs,k)
        return w0 + vsk

    def get_taylor_term(self, v, s, k, grad, grad_phi2=None, second_order=False):
        sigma_diag = torch.diag(s.view(-1))
        k = k.view(-1,1)
        vs = torch.matmul(torch.transpose(v, 0, 1), sigma_diag)
        vsk = self.sd_moved * torch.matmul(vs,k)

        if second_order == False:
            taylor_term = torch.matmul(grad, vsk)
        else:
            assert grad_phi2 != None, "Please input Hessian matrix"
            second_order_term = torch.matmul(torch.transpose(vsk, 0, 1), grad_phi2)
            second_order_term = torch.matmul(second_order_term, vsk)
            taylor_term = torch.matmul(grad,vsk) + 0.5*second_order_term
        return taylor_term

    def key_init_guess(self):
        return torch.zeros((self.key_len, 1), device=self.device)


    def calculate_classification_acc(self,approx_key):
        key_acc = torch.sum(approx_key == self.key)
        acc = key_acc / self.key_len
        return acc

    def make_image(self,tensor):
        return (
            tensor.detach()
                .clamp_(min=-1, max=1)
                .add(1)
                .div_(2)
                .mul(255)
                .type(torch.uint8)
                .permute(0, 2, 3, 1)
                .to("cpu")
                .numpy()
        )
    def penalty_1(self, latent, upper, lower):
        penalty1 = torch.sum(self.relu(latent-upper))
        penalty2 = torch.sum(self.relu(lower-latent))

        return penalty1+penalty2

    def noise_regularize(self, noises):
        loss = 0

        for noise in noises:
            size = noise.shape[2]

            while True:
                loss = (
                        loss
                        + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                        + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
                )

                if size <= 8:
                    break

                noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
                noise = noise.mean([3, 5])
                size //= 2

        return loss

    def latent_noise(self, latent, strength):
        noise = torch.randn_like(latent) * strength

        return latent + noise

    def get_lr(self, t, initial_lr, rampdown=0.25, rampup=0.05):
        lr_ramp = min(1, (1 - t) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / rampup)

        return initial_lr * lr_ramp

    def store_results(self, w0_reconstructed, wx_reconstructed, original_image, w0, wx, key):
        isExist = os.path.exists(self.save_dir)
        if not isExist:
            os.makedirs(self.save_dir)
        filename = self.save_dir + "normalized.pt"
        result_file = {
            "w0_reconstructed_img": w0_reconstructed,
            "wx_reconstructed_img": wx_reconstructed,
            "original_img": original_image,
            "w0": w0,
            "key": key,
            "wx" : wx
        }

        img_name = self.save_dir + "w0_reconstructed.png"
        pil_img = Image.fromarray(w0_reconstructed[0])
        pil_img.save(img_name)
        img_name = self.save_dir + "wx_reconstructed.png"
        pil_img = Image.fromarray(wx_reconstructed[0])
        pil_img.save(img_name)
        img_name = self.save_dir + "Target.png"
        pil_img = Image.fromarray(original_image[0])
        pil_img.save(img_name)
        torch.save(result_file, filename)


if __name__ == "__main__":
    optim = watermark_optimization()
    percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=optim.device.startswith("cuda"))
    sigma_64, v_cap, u_cap, pc, sigma_512, latent_mean, latent_std = optim.PCA()
    upper_limit = latent_mean+4*torch.transpose(torch.matmul(torch.transpose(sigma_512,0,1), pc),0,1)
    lower_limit = latent_mean-4*torch.transpose(torch.matmul(torch.transpose(sigma_512,0,1), pc),0,1)

    max_alpha, _ = torch.lstsq(upper_limit, torch.transpose(u_cap, 0, 1))  # solve for init of for alpha = [512x1] tensor
    max_alpha = max_alpha[0:optim.critical_point, :]  # solution for alpha = [448 x 1] tensor

    min_alpha, _ = torch.lstsq(lower_limit, torch.transpose(u_cap, 0, 1))  # solve for init of for alpha = [512x1] tensor
    min_alpha = min_alpha[0:optim.critical_point, :]  # solution for alpha = [448 x 1] tensor

    avg_alpha, _ = torch.lstsq(latent_mean, torch.transpose(u_cap, 0, 1))  # solve for init of for alpha = [512x1] tensor
    avg_alpha = avg_alpha[0:optim.critical_point, :]  # solution for alpha = [448 x 1] tensor

    target_img, target_w0, target_wx = optim.generate_target_image(sigma_64, v_cap)

    target_w0 = target_w0.detach()
    target_w0_t = torch.transpose(target_w0,0,1)

    v_cap_t = torch.transpose(v_cap, 0, 1)
    ata = torch.inverse(torch.matmul(v_cap, torch.transpose(v_cap, 0, 1)))
    projection_v = torch.matmul(torch.matmul(torch.matmul(v_cap_t, ata), v_cap), latent_mean)

    u_cap_t = torch.transpose(u_cap, 0, 1)
    ata = torch.inverse(torch.matmul(u_cap, torch.transpose(u_cap, 0, 1)))
    projection_u = torch.matmul(torch.matmul(torch.matmul(u_cap_t, ata), u_cap), latent_mean)

    a = projection_u + projection_v

    Beta, _ = torch.lstsq(projection_v, torch.transpose(v_cap, 0, 1))  # solve for init of for alpha = [512x1] tensor
    Beta = Beta[0:optim.key_len, :]  # solution for Beta = [64 x 1] tensor

    alpha, _ = torch.lstsq(projection_u, torch.transpose(u_cap, 0, 1))  # solve for init of for alpha = [512x1] tensor
    alpha = alpha[0:optim.critical_point, :]  # solution for alpha = [448 x 1] tensor

    gamma, _ = torch.lstsq(latent_mean, torch.transpose(pc, 0, 1))  # solve for init of for alpha = [512x1] tensor
    gamma = gamma[0:optim.style_space_dim, :]  # solution for alpha = [448 x 1] tensor

    max_gamma, _ = torch.lstsq(upper_limit, torch.transpose(pc, 0, 1))  # solve for init of for alpha = [512x1] tensor
    max_gamma = max_gamma[0:optim.style_space_dim, :]  # solution for alpha = [448 x 1] tensor

    min_gamma, _ = torch.lstsq(lower_limit, torch.transpose(pc, 0, 1))  # solve for init of for alpha = [512x1] tensor
    min_gamma = min_gamma[0:optim.style_space_dim, :]  # solution for alpha = [448 x 1] tensor
    use_alpha = True
    if use_alpha:
        gamma = alpha
        max_gamma = max_alpha
        min_gamma = min_alpha
        pc=u_cap
    w0_init = target_w0
    estimated_image = optim.generate_image(w0_init)
    loss = optim.get_loss(target_img, estimated_image, loss_func = 'perceptual')
    print(loss)
    key = optim.key_init_guess()
    key.requires_grad = True
    gamma.requires_grad = True
    sigmoid = torch.nn.Sigmoid()
    w0 = latent_mean.detach()
    w0.requires_grad = True
    Beta.requires_grad = True
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    lr = 0.1
    # noise_strength = 0.01
    optimizer = torch.optim.Adam([gamma, key], lr=lr)
    for i in tqdm(range(optim.steps)):
        t = i / optim.steps
        optim.g_ema.zero_grad()
        optimizer.zero_grad()
        w0 = torch.matmul(torch.transpose(pc, 0, 1), gamma)
        wx = optim.get_new_latent(v_cap, sigma_64, sigmoid(key), w0)
        # if i/optim.steps < 0.5:
        #     wx = optim.latent_noise(wx, noise_strength)

        # w0 = optim.latent_noise(w0, noise_strength)
        w0.retain_grad()
        estimated_image = optim.generate_image(wx)
        loss_1 = optim.get_loss(target_img, estimated_image, loss_func = "perceptual")
        loss_total = loss_1 + 0.1*optim.penalty_1(gamma,max_gamma,min_gamma)
        #loss_total = loss_1 + 100 * optim.penalty_1(alpha, max_alpha, min_alpha)
        loss_total.backward(retain_graph=True)
        optimizer.step()

        # optim.g_ema.zero_grad()
        # optimizer.zero_grad()
        # w0 = torch.matmul(torch.transpose(u_cap, 0, 1), alpha) + projection_v
        # w0.retain_grad()
        # estimated_image = optim.generate_image(w0)
        # loss_1 = optim.get_loss(target_img, estimated_image)
        # loss_total = loss_1 + 100*optim.penalty_1(latent_mean,upper_limit,lower_limit)
        # loss_total.backward(retain_graph=True)
        # optimizer.step()

        if (i+1) % 10 == 0:
            print("w0 loss: {}".format(loss_total.item()))
            print('cosine similarity of style vector: {}'
                  .format(cos(w0.view(-1), target_w0.view(-1))))
    #
    # optimizer = torch.optim.SGD([alpha], lr=0.01, momentum=0.9)
    # for i in tqdm(range(optim.steps)):
    #     optim.g_ema.zero_grad()
    #     optimizer.zero_grad()
    #     w0 = torch.matmul(torch.transpose(u_cap, 0, 1), alpha) + projection_v
    #     w0.retain_grad()
    #     estimated_image = optim.generate_image(w0)
    #     loss_1 = optim.get_loss(target_img, estimated_image)
    #     loss_total = loss_1 + 100*optim.penalty_1(latent_mean,upper_limit,lower_limit)
    #     loss_total.backward(retain_graph=True)
    #     optimizer.step()
    #     if (i+1) % 10 == 0:
    #         print("w0 loss: {}".format(loss_1.item()))
    #         print('cosine similarity of style vector: {}'
    #               .format(cos(w0.view(-1), target_w0.view(-1))))

    alpha = alpha.detach()
    print(optim.calculate_classification_acc(torch.round(sigmoid(key))))
    w0 = torch.matmul(torch.transpose(pc, 0, 1), gamma)
    # print(w0)
    # w0=torch.transpose(target_w0, 0, 1).detach()
    w0 = w0.detach()
    optimizer = torch.optim.Adam([key], lr=1)
    tanh = torch.nn.Tanh()
    step = 500
    for i in tqdm(range(step)):
        optim.g_ema.zero_grad()
        optimizer.zero_grad()
        wx = optim.get_new_latent(v_cap, sigma_64, sigmoid(key), w0)
        estimated_image = optim.generate_image(wx)
        loss_1 = optim.get_loss(target_img, estimated_image)
        loss_2 = optim.get_loss(target_img, estimated_image)
        loss_total = loss_1# + 100*optim.penalty_2(wx,upper_limit,lower_limit)
        loss_total.backward(retain_graph=True)
        optimizer.step()
        if (i+1) % 10 == 0:
            print("Taylor loss: {}, w0 loss: {}".format(loss_total.item(), loss_1.item()))
    wx = optim.get_new_latent(v_cap,sigma_64,sigmoid(key),w0)
    wx_estimated_image = optim.generate_image(wx)
    w0_reconstructed = optim.make_image(estimated_image)
    wx_reconstructed = optim.make_image(wx_estimated_image)
    original_image = optim.make_image(target_img)
    key_retrived = torch.round(sigmoid(key))
    print('cosine similarity of key:{}, \ncosine similarity of style vector: {}'
    .format(cos(key_retrived, optim.key), cos(wx.view(-1), target_wx.view(-1))))
    optim.store_results(w0_reconstructed, wx_reconstructed, original_image, w0, wx, sigmoid(key))

    print((key_retrived, optim.key))
    print(optim.calculate_classification_acc(torch.round(sigmoid(key))))


