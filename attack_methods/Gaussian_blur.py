import torchvision.transforms as T
import torch
import torch.nn as nn
import numpy as np
import random


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

##Uploading test
class Gaussian_blur(nn.Module):
    def __init__(self,kernel_size, is_train):
        super(Gaussian_blur, self).__init__()

        self.sigma = np.array(kernel_size) / 3.0
        self.filter_size = [3,9,15,19,25] #This filter size is calculated based on scipy source.
        if (not is_train):
            self.sigma = [self.sigma[-1]]
            self.filter_size = [self.filter_size[-1]]

    def forward(self, image):

        index = random.choice(np.arange(len(self.sigma)))

        filter_size = (self.filter_size[index], self.filter_size[index])
        sigma = (self.sigma[index], self.sigma[index])

        blurrer = T.GaussianBlur(filter_size, sigma=sigma).to(device)
        blurred_image = blurrer(image)

        return blurred_image

