import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
import numpy as np
import lpips
from model import Generator
import matplotlib.pyplot as plt

device = 'cuda:0'
ckpt = './checkpoint/550000.pt'
files = ['./sample_shifted_448/000018.png']
n_mean_latent = 10000 # num of style vector to sample
img_size = 256 # image size
style_space_dim = 512
mapping_network_layer = 8
resize = min(img_size, 256)
critical_point = 448 # 512 - 64, num of high var pc
use_mse_loss = True

transform = transforms.Compose(
    [
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
# Get batched image file
imgs = []
for imgfile in files:
    img = transform(Image.open(imgfile).convert("RGB"))
    imgs.append(img)
imgs = torch.stack(imgs, 0).to(device)

# Get generator
g_ema = Generator(img_size, style_space_dim, mapping_network_layer)
g_ema.load_state_dict(torch.load(ckpt)["g_ema"], strict=False) # load ckpt
g_ema.eval() # set to eval mode
g_ema = g_ema.to(device) # push to device


latent_in = np.load('./projection_Test/latent_in.npy')
latent_in = torch.tensor(latent_in, device = device)
img_gen, _ = g_ema([latent_in], input_is_latent=True)

loss = F.mse_loss(img_gen, imgs)

print(loss)




def make_image(tensor):
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

trials = 'trial_1_original_3/'
test_output_dir = "./projection_Test/" + trials
isExist = os.path.exists(test_output_dir)
if not isExist:
    os.makedirs(test_output_dir)

filename = "./projection_Test/" + trials + "normalized.pt"

img_ar = make_image(img_gen)

result_file = {}
for i, input_name in enumerate(files):

    result_file[input_name] = {
        "img": img_gen[i],
        "latent": latent_in[i]
    }
    img_name = "./projection_Test/"+trials+os.path.splitext(os.path.basename(input_name))[0] + "-project.png"
    pil_img = Image.fromarray(img_ar[i])
    pil_img.save(img_name)
torch.save(result_file, filename)

