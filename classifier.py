import os
import math
import torch
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm
import lpips
from model import Generator
import torchvision.transforms as T
import scipy.stats.qmc as scipy_stats
import time
import numpy as np

class watermark_optimization:
    def __init__(self):
        # Define hyper parameter
        np.random.seed(2022)
        # for cuda
        self.device = 'cuda:0'
        self.ckpt = './checkpoint/550000.pt'
        self.n_mean_latent = 10000  # num of style vector to sample
        self.steps = 1000 # Num steps for optimizing
        self.img_size = 256  # image size
        self.style_space_dim = 512
        self.mapping_network_layer = 8
        self.critical_point = 448  # 512 - 64, num of high var pc
        self.sd_moved = 6  # How many standard deviation to move
        self.lr = 0.2
        self.key_len = self.style_space_dim - self.critical_point
        self.save_dir = './projection_Test/trials_alpha_mean_final/'
        self.relu = torch.nn.ReLU()
        # Get generator
        g_ema = Generator(self.img_size, self.style_space_dim, self.mapping_network_layer)
        g_ema.load_state_dict(torch.load(self.ckpt)["g_ema"], strict=False)  # load ckpt
        g_ema.eval()  # set to eval mode
        self.g_ema = g_ema.to(self.device)  # push to device



    def get_loss(self, img1, img2, loss_func='mse'):
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

    def key_init_guess(self):
        """init guess for key, all zeros (before entering sigmoid function)"""
        return torch.zeros((self.key_len, 1), device=self.device)

    def calculate_classification_acc(self, approx_key, target_key):
        """Calculate digit-wise key classification accuracy"""
        key_acc = torch.sum(approx_key == target_key)
        acc = key_acc / self.key_len
        return acc

    def make_image(self, tensor):
        """Image postprocessing for output"""
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

    def get_image(self, tensor):
        """Image postprocessing for output"""
        return (tensor.detach()
                .mul(2)
                 .sub(1)
        )


    def penalty_1(self, latent, upper, lower):
        """penalty for alpha that exceed the boundary"""
        penalty1 = torch.sum(self.relu(latent - upper))
        penalty2 = torch.sum(self.relu(lower - latent))

        return penalty1 + penalty2

    def store_results(self, w0_reconstructed, wx_reconstructed,  w0, wx, key,iter=None):
        if iter == None:
            pass
        else:
            self.save_dir_i = self.save_dir + 'test_{}/'.format(iter)
        isExist = os.path.exists(self.save_dir_i)
        if not isExist:
            os.makedirs(self.save_dir_i)
        filename = self.save_dir_i + "normalized.pt"
        result_file = {
            "w0_reconstructed_img": w0_reconstructed,
            "wx_reconstructed_img": wx_reconstructed,
            "w0": w0,
            "key": key,
            "wx": wx
        }

        img_name = self.save_dir_i + "w0_reconstructed.png"
        pil_img = Image.fromarray(w0_reconstructed[0])
        pil_img.save(img_name)
        img_name = self.save_dir_i + "wx_reconstructed.png"
        pil_img = Image.fromarray(wx_reconstructed[0])
        pil_img.save(img_name)
        torch.save(result_file, filename)

    def get_noise(self):
        rng = np.random.default_rng(seed=2002)
        log_size = int(math.log(self.img_size, 2))

        noises = [torch.tensor(rng.standard_normal((1, 1, 2 ** 2, 2 ** 2)), dtype=torch.float32, device=self.device)]

        for i in range(3, log_size + 1):
            for _ in range(2):
                noises.append(torch.tensor(np.random.standard_normal((1, 1, 2 ** i, 2 ** i)), dtype=torch.float32,
                                           device=self.device))

        return noises

if __name__ == "__main__":

    optim = watermark_optimization()

    start = time.time() # count times to complete
    pca = torch.load("./test_images/pca.pt")
    target_image_raw = torch.load("./test_images/target_img_total.pt")

    sigma_64 = pca['sigma_64'].to(device=optim.device).view(-1,1)
    sigma_512 = pca['sigma_512'].to(device=optim.device).view(-1,1)
    v_cap = pca['v_cap'].to(device=optim.device)
    u_cap = pca['u_cap'].to(device=optim.device)
    latent_mean = pca['latent_mean'].to(device=optim.device)

    # Get projections of the latent mean(for initial guess)
    v_cap_t = torch.transpose(v_cap, 0, 1)
    ata = torch.inverse(torch.matmul(v_cap, torch.transpose(v_cap, 0, 1)))
    projection_v = torch.matmul(torch.matmul(torch.matmul(v_cap_t, ata), v_cap), latent_mean)

    u_cap_t = torch.transpose(u_cap, 0, 1)
    ata = torch.inverse(torch.matmul(u_cap, torch.transpose(u_cap, 0, 1)))
    projection_u = torch.matmul(torch.matmul(torch.matmul(u_cap_t, ata), u_cap), latent_mean)
    sigma_448 = sigma_512[0:optim.critical_point, :]

    # Get the boundary of alpha
    alpha_bar, _ = torch.lstsq(projection_u,
                               torch.transpose(u_cap, 0, 1))  # solve for init of for alpha = [512x1] tensor
    alpha_bar = alpha_bar[0:optim.critical_point, :]  # solution for alpha = [448 x 1] tensor
    max_alpha = alpha_bar + 3 * sigma_448
    min_alpha = alpha_bar - 3 * sigma_448


    noise = optim.get_noise()
    tests = 10  # Number of image tests
    early_termination = 0.0002  # Terminate the optimization if loss is below this number
    success = 0  # count number of success
    acc_total = []
    # Import perceptual loss, cosine similarity and sigmoid function
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    sigmoid = torch.nn.Sigmoid()
    # Import Latin Hypercube Sampling method
    samlping = scipy_stats.LatinHypercube(d=optim.critical_point, centered=True)
    target_k_total = torch.load("./test_images/store_key.pt")
    target_wx_total = torch.load("./test_images/store_wx.pt")
    percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=optim.device.startswith("cuda"))
    for iter in range(tests):
        loss = []
        a = []
        k = []
        image = Image.open('./test_images/perturbed_image/target_wx_{}.png'.format(iter)).convert('RGB')
        transform = T.ToTensor()
        tensor = transform(image)
        target_img = optim.get_image(tensor).to(optim.device)
        target_k = target_k_total[iter].to(optim.device)
        target_k = target_k.view(-1,1)
        target_wx = target_wx_total[iter].to(optim.device)
        sample = samlping.random(n=20)  # Sample init guesses
        sample = torch.tensor(sample, dtype=torch.float32, device=optim.device).detach()
        for alpha in sample:
            lr_decay_rate = 4
            lr_segment = lr_decay_rate - 1
            alpha = alpha.view(-1, 1)
            alpha = 2 * torch.multiply(alpha, sigma_448) - 1 * sigma_448 + alpha_bar
            alpha.requires_grad = True
            key = optim.key_init_guess()
            key.requires_grad = True
            optimizer = torch.optim.Adam([alpha,key],lr=optim.lr)

            lr = optim.lr
            for i in tqdm(range(optim.steps)):
                optim.g_ema.zero_grad()
                optimizer.zero_grad()
                w0 = torch.matmul(torch.transpose(u_cap, 0, 1), alpha)
                wx = optim.get_new_latent(v_cap, sigma_64, sigmoid(key), w0)
                estimated_image = optim.generate_image(wx, noise)
                loss_1 = optim.get_loss(target_img, estimated_image, loss_func="perceptual")
                loss_total = loss_1 + 0.1 * optim.penalty_1(alpha, max_alpha, min_alpha)
                # Early termination if init guess not promising
                if i > optim.steps / 4:
                    if loss_total > 0.3:
                        break
                if i > optim.steps / 2:
                    if loss_total > 0.2:
                        break

                # Discrete learning rate decay
                if (i + 1) % int(optim.steps / lr_decay_rate) == 0:
                    lr = lr_segment * optim.lr / lr_decay_rate
                    lr_segment -= 1
                    optimizer.param_groups[0]["lr"] = lr
                    print(lr)
                if loss_total <= early_termination:
                    break

                loss_total.backward()
                optimizer.step()

                if (i + 1) % 10 == 0:
                    print("w0 loss: {}".format(loss_total.item()))
                    print('cosine similarity of style vector: {}'
                          .format(cos(wx.view(-1), target_wx.view(-1))))

            if loss_total <= early_termination:
                break
            else:
                loss.append(loss_total.item())
                a.append(alpha)
                k.append(key)

        # If early terminated, pick the last one, else, pick the one with min loss
        if loss_total <= early_termination:
            pass
        else:
            min_item = min(loss)
            index = loss.index(min_item)
            alpha = a[index]
            key = k[index]

        alpha = alpha.detach()
        # print(optim.calculate_classification_acc(torch.round(sigmoid(key))))
        w0 = torch.matmul(torch.transpose(u_cap, 0, 1), alpha)
        w0 = w0.detach()
        estimated_w0 = optim.generate_image(w0, noise)
        # optimizer = torch.optim.Adam([key], lr=1)
        # step = 500
        # for i in tqdm(range(step)):
        #     optim.g_ema.zero_grad()
        #     optimizer.zero_grad()
        #     wx = optim.get_new_latent(v_cap, sigma_64, sigmoid(key), w0)
        #     estimated_image = optim.generate_image(wx,noise)
        #     loss = optim.get_loss(target_img, estimated_image)
        #     loss.backward(retain_graph=True)
        #     optimizer.step()
        #     if (i+1) % 10 == 0:
        #         print("Taylor loss: {}, w0 loss: {}".format(loss.item(), loss.item()))

        # Make images for inspection
        wx = optim.get_new_latent(v_cap, sigma_64, sigmoid(key), w0)
        wx_estimated_image = optim.generate_image(wx, noise)
        w0_reconstructed = optim.make_image(estimated_w0)
        wx_reconstructed = optim.make_image(wx_estimated_image)
        key_retrived = torch.round(sigmoid(key))
        print('cosine similarity of key:{}, \ncosine similarity of style vector: {}'
              .format(cos(key_retrived, target_k), cos(wx.view(-1), target_wx.view(-1))))
        optim.store_results(w0_reconstructed, wx_reconstructed, w0, wx,
                            sigmoid(key), iter)
        acc = optim.calculate_classification_acc(torch.round(sigmoid(key)),target_k)
        # print((key_retrived, optim.key))
        print(acc)
        acc_total.append(acc)
        if acc == 1.0:
            success += 1
        print('Among {} tests, success rate is: {}'.format(iter + 1, success / (iter + 1)))
        end = time.time()
        print('time taken for optimization:', end - start)
        with open(optim.save_dir + 'listfile.txt', 'w') as filehandle:
            for listitem in acc_total:
                filehandle.write('%s\n' % listitem)

