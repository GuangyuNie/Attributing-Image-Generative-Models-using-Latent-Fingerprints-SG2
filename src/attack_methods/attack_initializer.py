from .Gaussian_blur import Gaussian_blur
from .Gaussian_noise import Gaussian_noise
#from .Jpeg_compression import JpegCompression
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
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        attack = DiffJPEG(height=opt.img_size, width=opt.img_size, differentiable=is_train, quality=opt.jpeg_quality).to(device)
    elif (attack_method == 'Combination'):
        attacks = []
        attacks.append(Gaussian_blur(sigma=[opt.blur_sigma], is_train = is_train))
        attacks.append(Gaussian_noise([opt.noise_sigma], is_train))
        attacks.append(DiffJPEG(height=opt.image_size, width=opt.image_size, differentiable=is_train, quality=opt.jpeg_quality))

        attack = Combination_attack(attacks, is_train)


    else:
        raise ValueError("Not available Attacks")



    return attack