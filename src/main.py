import os
import math
import torch
from PIL import Image
from tqdm import tqdm
import custom_lpips
from torch.nn import functional as F
import time
import numpy as np
from attack_methods import attack_initializer
import scipy.stats.qmc as scipy_stats
import datetime

from params import opt
from PCA import GetPCA
from generator import GetGen
from utils import *
def make_dir(sigma,shift):
    save_dir = opt.save_dir + "{}/fixed_sigma_{}/shift_{}/".format(opt.augmentation, sigma, shift).replace(
        '.', '')
    return save_dir

def get_alpha_bound(sigma_512,shift):
    max_alpha = 3 * sigma_512
    min_alpha = -3 * sigma_512
    max_alpha = torch.cat([max_alpha[0:shift, :], max_alpha[shift + opt.key_len:generator.style_space_dim, :]], dim=0)
    min_alpha = torch.cat([min_alpha[0:shift, :], min_alpha[shift + opt.key_len:generator.style_space_dim, :]], dim=0)
    return max_alpha,min_alpha

def get_uv(shift,pc,sigma_64,sigma_512):

    v_cap = torch.tensor(pc[shift:shift + opt.key_len, :], dtype=torch.float32,
                         device=opt.device)  # low var pc [64x512]
    u_cap = torch.cat([pc[0:shift, :], pc[shift + opt.key_len:generator.style_space_dim, :]], dim=0)
    u_cap_t = torch.transpose(u_cap, 0, 1)
    sigma_64 = fixed_sigma * torch.ones_like(sigma_64)
    sigma_448 = torch.cat([sigma_512[0:shift, :], sigma_512[shift + opt.key_len:generator.style_space_dim, :]], dim=0)
    sigma_448 = sigma_448.repeat(1, opt.batch_size)
    sigma_64 = sigma_64.repeat(1, opt.batch_size)
    return sigma_64,sigma_448,u_cap,u_cap_t,v_cap

def get_lr(iter):
    return opt.lr * math.exp(-0.001 * (iter + 1))

def optimization(target_img):
    sample = samlping.random(n=opt.n)  # Sample init guesses
    sample = torch.tensor(sample, dtype=torch.float32, device=opt.device).detach()
    for alpha in sample:
        lr_decay_rate = 4
        lr_segment = lr_decay_rate - 1
        alpha = alpha.view(-1, 1)
        alpha = 2 * torch.multiply(alpha, sigma_448) - 1 * sigma_448
        alpha.requires_grad = True
        key = key_init_guess()
        key.requires_grad = True
        optimizer = torch.optim.Adam([alpha, key], lr=opt.lr)
        early_terminate = False # todo delete this
        for i in tqdm(range(opt.steps)):
            generator.g_ema.zero_grad()
            optimizer.zero_grad()
            w0 = torch.matmul(torch.transpose(u_cap, 0, 1), alpha) + generator.latent_mean
            wx = generator.get_new_latent(v_cap, sigma_64, sigmoid(key), w0)
            estimated_image = generator.generate_image(wx, noise)
            loss_1 = get_loss(target_img, estimated_image, loss_func="perceptual")

            loss_total = loss_1 + 0.1 * alpha_bound(alpha, max_alpha, min_alpha)

            optimizer.param_groups[0]["lr"] = get_lr(i)

            loss_total.backward()
            optimizer.step()
            l2 = torch.dist(w0.view(-1), target_w0.view(-1), p=2)
            acc = calculate_classification_acc(torch.round(sigmoid(key)), generator.key)

            if (i + 1) % 100 == 0:
                print("Perceptual loss: {:.6f}".format(loss_total.item()))
                print('bit-wise acc: {:.4f}'.format(acc))

        acc = calculate_classification_acc(torch.round(sigmoid(key)), generator.key)
        if acc == 1:  # Todo: Delete all early termination methods
            early_terminate = True
            break
        else:
            loss.append(loss_total.item())
            a.append(alpha)
            k.append(key)

        # If early terminated, pick the last one, else, pick the one with min loss
        if early_terminate == True:
            pass
        else:
            min_item = min(loss)
            index = loss.index(min_item)
            alpha = a[index]
            key = k[index]
    acc = calculate_classification_acc(torch.round(sigmoid(key)), generator.key)
    return alpha, key, acc



if __name__ == "__main__":
    # torch.manual_seed(1346)

    get_pca = GetPCA()
    generator = GetGen()
    fixed_sigma = opt.sigma
    shift = opt.shift

    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    sigmoid = torch.nn.Sigmoid()
    samlping = scipy_stats.LatinHypercube(d=generator.num_main_pc, centered=True)

    save_dir = save_config(make_dir(fixed_sigma,shift))



    sigma_64,sigma_448,u_cap,u_cap_t,v_cap= get_uv(shift,generator.pc,generator.sigma_64,generator.sigma_512)
    # Get the boundary of alpha
    max_alpha, min_alpha = get_alpha_bound(generator.sigma_512,shift)
    noise = get_noise()

    tests = opt.sample_size  # Number of image tests
    early_termination = 0.0005  # Terminate the optimization if loss is below this number #todo: delete this
    acc_total = []
    success = 0  # count number of success
    # Import Latin Hypercube Sampling method
    for iter in range(tests):
        loss = []
        a = []
        k = []
        rand_alpha = torch.multiply(sigma_448, torch.randn((generator.num_main_pc, opt.batch_size),device=opt.device))
        target_img, target_w0, target_wx, true_key = generator.generate_with_alpha(rand_alpha, u_cap_t, sigma_64, v_cap, noise)
        target_img = generator.augmentation(target_img).detach()
        alpha, key, acc = optimization(target_img)

        print('sample: {}, attribution accuracy: {}'.format(iter,acc))

        target_w0_img = generator.generate_image(target_w0, noise)
        target_w0_img = make_image(target_w0_img)

        target_wx_img = generator.generate_image(target_wx, noise)
        targe_wx_img = make_image(target_img)

        perturbed = generator.augmentation(target_wx_img)
        perturbed = make_image(perturbed)

        # watermark_pos = np.uint8((np.int16(targe_wx_img) - np.int16(target_w0_img)).clip(0, 255))
        # watermark_neg = np.uint8((np.int16(target_w0_img) - np.int16(targe_wx_img)).clip(0, 255))
        #
        # watermark_pos = np.uint8(watermark_pos)
        # watermark_neg = np.uint8(watermark_neg)

        store_results(save_dir, iter, target_w0_img, targe_wx_img, perturbed)
        acc_total.append(acc)
        if acc == 1.0:
            success += 1
        classification_acc = success / (iter + 1)
        # print('Among {} tests, success rate is: {}'.format(iter + 1, classification_acc))
        # print('time taken for optimization:', end - start)
        with open(save_dir + 'result.txt', 'w') as filehandle:
            for i, listitem in enumerate(acc_total):
                filehandle.write('\n sample index: {}, bit acc: {}, attribution acc: {}'.format(i, listitem.item(),
                                                                                                classification_acc))
