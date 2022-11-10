import os
import math
import torch
from PIL import Image
from torch.nn import functional as F
import numpy as np

from params import opt
import custom_lpips

import datetime
import yaml



percept = custom_lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=opt.device)
relu = torch.nn.ReLU()

def key_init_guess():
    """init guess for key, all zeros (before entering sigmoid function)"""
    return torch.zeros((opt.key_len, 1), device=opt.device)

def save_config(save_dir):
    """standard saving module, input save_dir, create folder and save the config, and return time-varient save dir"""
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  # get time
    save_dir = os.path.join(save_dir, now, '')
    sampling_conf = vars(opt)
    isExist = os.path.exists(save_dir)
    if not isExist:
        os.makedirs(save_dir)
    sampling_file = os.path.join(save_dir, "sampling_config.yaml")
    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    return save_dir


def calculate_classification_acc(approx_key, target_key):
    """Calculate digit-wise key classification accuracy"""
    key_acc = torch.sum(target_key == approx_key)
    acc = key_acc / opt.key_len
    return acc


def get_loss(img1, img2, loss_func='perceptual'):
    """Loss function, default: MSE loss"""
    if loss_func == "mse":
        loss = F.mse_loss(img1, img2)
    elif loss_func == "perceptual":
        loss = percept(img1, img2)
    return loss


def alpha_bound(latent, upper, lower):
    """penalty for alpha that exceed the boundary"""
    penalty1 = torch.sum(relu(latent - upper))
    penalty2 = torch.sum(relu(lower - latent))

    return penalty1 + penalty2


def make_image(tensor,get_torch=False):
    """Image postprocessing for output"""
    if get_torch:
        return(tensor.detach()
            .clamp_(min=-1, max=1)
            .add(1)
            .div_(2)
            .mul(255)
            .round()
            .type(torch.uint8))
    else:
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

def store_results(save_dir,iter,original_image_w0=None, original_image_wx=None,watermark_pos=None,watermark_neg=None,):
    store_path_w0 = 'original/'
    store_path_wx = 'watermarked/'
    watermark_pos_path = 'watermark_pos/'
    watermark_neg_path = 'watermark_neg/'

    if original_image_w0 is not None:
        isExist = os.path.exists(save_dir + store_path_w0)
        if not isExist:
            os.makedirs(save_dir + store_path_w0)
    if original_image_wx is not None:
        isExist = os.path.exists(save_dir + store_path_wx)
        if not isExist:
            os.makedirs(save_dir + store_path_wx)
    if watermark_pos is not None:
        isExist = os.path.exists(save_dir + watermark_pos_path)
        if not isExist:
            os.makedirs(save_dir + watermark_pos_path)
    if watermark_neg is not None:
        isExist = os.path.exists(save_dir + watermark_neg_path)
        if not isExist:
            os.makedirs(save_dir + watermark_neg_path)


    for i in range(opt.batch_size):
        if original_image_w0 is not None:
            img_name = save_dir + store_path_w0 + f'{iter:06d}.png'
            pil_img = Image.fromarray(original_image_w0[i])
            pil_img.save(img_name)
        if original_image_wx is not None:
            img_name = save_dir + store_path_wx + f'{iter:06d}.png'
            pil_img = Image.fromarray(original_image_wx[i])
            pil_img.save(img_name)
        if watermark_pos is not None:
            img_name = save_dir + watermark_pos_path + f'{iter:06d}.png'
            pil_img = Image.fromarray(watermark_pos[i])
            pil_img.save(img_name)
        if watermark_neg is not None:
            img_name = save_dir + watermark_neg_path + f'{iter:06d}.png'
            pil_img = Image.fromarray(watermark_neg[i])
            pil_img.save(img_name)


def get_noise():
    rng = np.random.default_rng(seed=2002)
    log_size = int(math.log(opt.img_size, 2))

    noises = [torch.tensor(rng.standard_normal((1, 1, 2 ** 2, 2 ** 2)), dtype=torch.float32, device=opt.device)]

    for i in range(3, log_size + 1):
        for _ in range(2):
            noises.append(torch.tensor(np.random.standard_normal((1, 1, 2 ** i, 2 ** i)), dtype=torch.float32,
                                       device=opt.device))
    return noises
