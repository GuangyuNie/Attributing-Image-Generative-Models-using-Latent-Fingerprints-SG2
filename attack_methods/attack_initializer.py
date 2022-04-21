from .Crop import Crop
from .Gaussian_blur import Gaussian_blur
from .Gaussian_noise import Gaussian_noise
from .Jpeg_compression import JpegCompression
from .Combination import Combination_attack
import torch

def attack_initializer(attack_method, is_train):

    if (attack_method == 'Crop'):
        attack = Crop([0.8, 1], is_train)

    elif (attack_method == 'Noise'):
        attack = Gaussian_noise([0, 0.3], is_train)

    elif (attack_method == 'Blur'):
        #terminology would be different kernel_size
        attack = Gaussian_blur(kernel_size=[1,3,5,7,9], is_train = is_train)

    elif(attack_method == "Jpeg"):
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        attack = JpegCompression(device)

    elif (attack_method == 'Combination'):
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        attacks = []

        attacks.append(Gaussian_blur(kernel_size=[1,3,5,7,9], is_train = is_train))
        attacks.append(Crop([0.8, 1], is_train))
        attacks.append(Gaussian_noise([0, 0.1], is_train))
        attacks.append(JpegCompression(device))

        attack = Combination_attack(attacks, is_train)

    elif (attack_method == 'Combination_with_pillow'):
        # Combination Attack but Jpeg will be done after 3 attacks finished samples
        # img -> 3 attack -> Save PNG -> Pillow 75 -> Load
        attacks = []
        attacks.append(Gaussian_blur(kernel_size=[1, 3, 5, 7, 9], is_train=is_train))
        attacks.append(Crop([0.8, 1], is_train))
        attacks.append(Gaussian_noise([0, 0.1], is_train))

        attack = Combination_attack(attacks, is_train)

    elif (attack_method == "Identity"):
        attack = torch.nn.Identity()

    else:
        raise ValueError("Not available Attacks")



    return attack