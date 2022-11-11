from .Gaussian_blur import Gaussian_blur
from .Gaussian_noise import Gaussian_noise
from .Jpeg_compression import Jpeg
from .DiffJPEG_master.DiffJPEG import DiffJPEG
from .Combination import Combination_attack
import torch

import sys
sys.path.append("..")
from params import opt

def attack_initializer(attack_method, is_train):

    if (attack_method == 'Noise'):
        attack = Gaussian_noise([opt.noise_sigma], is_train)

    elif (attack_method == 'Blur'):
        attack = Gaussian_blur(sigma=[opt.blur_sigma], is_train = is_train)
    elif(attack_method == "Jpeg"):
        attack = Jpeg(is_train, opt.jpeg_quality, opt.img_size).to(opt.device)
    elif (attack_method == 'Combination'):
        attacks = []
        attacks.append(Gaussian_blur(sigma=[opt.blur_sigma], is_train = is_train))
        attacks.append(Gaussian_noise([opt.noise_sigma], is_train))
        attacks.append(Jpeg(is_train, opt.jpeg_quality, opt.img_size).to(opt.device))

        attack = Combination_attack(attacks, is_train)


    else:
        raise ValueError("Not available Attacks")



    return attack