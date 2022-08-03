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
import argparse

from attack_methods import attack_initializer



class watermark_optimization:
    def __init__(self):
        # Define hyper parameter
        self.device_ids = 0
        self.device = 'cuda:{}'.format(self.device_ids)
        self.ckpt = args.ckpt
        self.n_mean_latent = 10000  # num of style vector to sample
        self.steps = args.steps  # Num steps for optimizing
        self.img_size = args.img_size  # image size
        self.key_len = args.key_len
        self.style_space_dim = 512
        self.num_main_pc = self.style_space_dim - self.key_len
        self.mapping_network_layer = 8
        self.sd_moved = args.sd  # How many standard deviation to move
        self.lr = 0.2
        self.save_dir = args.save_dir
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
        k = k.view(-1, 1)
        vs = torch.transpose(v, 0, 1)
        vsk = s*self.sd_moved * torch.matmul(vs, k)
        return w0 + vsk

    def key_init_guess(self):
        """init guess for key, all zeros (before entering sigmoid function)"""
        return torch.zeros((self.key_len, 1), device=self.device)

    def calculate_classification_acc(self, approx_key, target_key):
        """Calculate digit-wise key classification accuracy"""
        key_acc = torch.sum(approx_key == target_key)
        acc = key_acc / self.key_len
        return acc

    # def latent_noise(self,latent, strength):
    #     noise = torch.randn_like(latent) * strength
    #     return latent + noise

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
                .type(torch.float32)
                )

    def penalty_1(self, latent, upper, lower):
        """penalty for alpha that exceed the boundary"""
        penalty1 = torch.sum(self.relu(latent - upper))
        penalty2 = torch.sum(self.relu(lower - latent))

        return penalty1 + penalty2


    def store_results(self, w0_reconstructed, wx_reconstructed, w0, wx, key, iter=None,shift=None):
        if iter == None:
            pass
        else:
            self.save_dir_i = self.save_dir  + 'shift_{}/'.format(shift)+ 'test_{}/'.format(iter)
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
    def augmentation(self, target_img,strength):
        """Image augmentation, default is None"""
        if args.augmentation != "None":
            attack = attack_initializer.attack_initializer(args.augmentation,strength,is_train=False)
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
        "--steps", type=int, default=2000, help="Number of optimization steps"
    )
    parser.add_argument(
        "--sd", type=int, default=6, help="Standard deviation moved"
    )
    parser.add_argument(
        "--n", type=int, default=20, help="Number of samples for Latin hypercube sampling method"
    )
    parser.add_argument(
        "--num_tests", type=int, default=100, help="Number of tests, plz make sure you have enough test samples"
    )

    parser.add_argument(
        "--key_len", type=int, default=64, help="Number of digit for the binary key"
    )

    parser.add_argument(
        "--save_dir", type=str, default='./result_image/', help="Directory for image output and acc result"
    )

    parser.add_argument(
        "--test_dir", type=str, default='./test_images/', help="Directory for image output and acc result"
    )

    parser.add_argument(
        "--augmentation", type=str, default='None', help="Augmentation method: Crop, Noise, Blur, Jpeg, Combination "
    ) # Todo: Delete this parse when publish
    args = parser.parse_args()
    shifts = [1,64,128,192,256,320,384,448]
    fixed_sigma = 0.1
    args.test_dir = args.test_dir + "key_{}_sd_{}_aug_{}/".format(args.key_len, args.sd,
                                                                  args.augmentation)
    args.save_dir = args.save_dir + "key_{}_sd_{}_aug_{}/".format(args.key_len, args.sd,
                                                                  args.augmentation)
    for shift in shifts:
        optim = watermark_optimization()
        start = time.time()  # count times to complete
        pca = torch.load(args.test_dir + 'shift_{}/'.format(shift) + "pca.pt")
        target = torch.load(args.test_dir + 'shift_{}/'.format(shift) + 'test_data.pt')

        sigma_64 = pca['sigma_64'].to(device=optim.device).view(-1, 1)
        sigma_512 = pca['sigma_512'].to(device=optim.device).view(-1, 1)
        latent_mean = pca['latent_mean'].to(device=optim.device)
        v_cap = pca['v_cap'].to(device=optim.device)
        u_cap = pca['u_cap'].to(device=optim.device)

        target_k_total = target['key']
        target_wx_total = target['wx']
        target_w0_total = target['w0']

        # Get projections of the latent mean(for initial guess)
        v_cap_t = torch.transpose(v_cap, 0, 1)
        ata = torch.inverse(torch.matmul(v_cap, torch.transpose(v_cap, 0, 1)))
        projection_v = torch.matmul(torch.matmul(torch.matmul(v_cap_t, ata), v_cap), latent_mean)

        u_cap_t = torch.transpose(u_cap, 0, 1)
        ata = torch.inverse(torch.matmul(u_cap, torch.transpose(u_cap, 0, 1)))
        projection_u = torch.matmul(torch.matmul(torch.matmul(u_cap_t, ata), u_cap), latent_mean)
        sigma_448 = torch.cat([sigma_512[0:shift, :],sigma_512[shift+64:optim.style_space_dim, :]],dim=0)

        # Get the boundary of alpha
        alpha_bar = torch.zeros((512,1)).to(optim.device)  # solve for init of for alpha = [512x1] tensor
        max_alpha = alpha_bar + 3 * sigma_512
        min_alpha = alpha_bar - 3 * sigma_512

        max_alpha = torch.cat([max_alpha[0:shift, :],max_alpha[shift+64:optim.style_space_dim, :]],dim=0)
        min_alpha = torch.cat([min_alpha[0:shift, :],min_alpha[shift+64:optim.style_space_dim, :]],dim=0)

        noise = optim.get_noise()
        tests = args.num_tests  # Number of image tests
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
            image = Image.open(args.test_dir + 'shift_{}/'.format(shift) + 'perturbed_image/target_wx_{}.png'.format(iter)).convert('RGB')
            transform = T.ToTensor()
            tensor = transform(image)
            target_img = optim.get_image(tensor).to(optim.device)
            target_k = target_k_total[iter].to(optim.device)
            target_k = target_k.view(-1, 1)
            target_wx = target_wx_total[iter].to(optim.device)
            target_w0 = target_w0_total[iter].to(optim.device)
            sample = samlping.random(n=args.n)  # Sample init guesses
            sample = torch.tensor(sample, dtype=torch.float32, device=optim.device).detach()
            for alpha in sample:
                # conversion_error = torch.zeros(3, 256, 256).to(optim.device)
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
                    wx = optim.get_new_latent(v_cap, fixed_sigma, sigmoid(key), w0)
                    estimated_image = optim.generate_image(wx, noise)
                    # estimated_image = rotation.rot_img(estimated_image, theta, dtype=torch.float32)
                    loss_1 = optim.get_loss(target_img, estimated_image, loss_func="perceptual")
                    loss_total = loss_1 + 0.1 * optim.penalty_1(alpha, max_alpha, min_alpha)
                    # if i > optim.steps / 4 and loss_total > 0.2:
                    #     break
                    # if i > optim.steps / 2 and loss_total > 0.1:
                    #     break
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
            w0 = torch.matmul(torch.transpose(u_cap, 0, 1), alpha)
            w0 = w0.detach()
            estimated_w0 = optim.generate_image(w0, noise)
            wx = optim.get_new_latent(v_cap, fixed_sigma, sigmoid(key), w0)
            wx_estimated_image = optim.generate_image(wx, noise)
            w0_reconstructed = optim.make_image(estimated_w0)
            wx_reconstructed = optim.make_image(wx_estimated_image)
            key_retrived = torch.round(sigmoid(key))
            print('cosine similarity of key:{}, \ncosine similarity of style vector: {}'
                  .format(cos(key_retrived, target_k), cos(wx.view(-1), target_wx.view(-1))))
            optim.store_results(w0_reconstructed, wx_reconstructed, w0, wx,
                                sigmoid(key), iter,shift)
            acc = optim.calculate_classification_acc(torch.round(sigmoid(key)), target_k)
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
