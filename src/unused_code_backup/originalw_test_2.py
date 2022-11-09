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

# torch.manual_seed(2001)
device = 'cuda:0'
ckpt = './checkpoint/550000.pt'
files = ['./sample_shifted_448/000018.png']
n_mean_latent = 10000 # num of style vector to sample
img_size = 256 # image size
style_space_dim = 512
mapping_network_layer = 8
resize = min(img_size, 256)
critical_point = 448 # 512 - 64, num of high var pc
len_key = style_space_dim - critical_point
eig_index = 0  # Index shifted for the last 64 digits (0~63)
sd_moved = 6  # How many standard deviation to move
use_mse_loss = True
shift = 0
step = 500
k = torch.zeros((len_key,1), device=device)
k.requires_grad=True
optimizer = optim.Adam([k], lr=1/sd_moved)

# transform = transforms.Compose(
#     [
#         transforms.Resize(resize),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#     ]
# )
# # Get batched image file
# imgs = []
# for imgfile in files:
#     img = transform(Image.open(imgfile))
#     imgs.append(img)
# imgs = torch.stack(imgs, 0).to(device)


# Get generator
g_ema = Generator(img_size, style_space_dim, mapping_network_layer)

g_ema.load_state_dict(torch.load(ckpt)["g_ema"], strict=False) # load ckpt
g_ema.eval() # set to eval mode
g_ema = g_ema.to(device) # push to device
# Do PCA
def PCA_test(n_mean_latent):
    pca = PCA()
    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device) # get a bunch of Z
        latent_out = g_ema.style(noise_sample) # get style vector from Z
        latent_out = latent_out.detach().cpu().numpy()
        pca.fit(latent_out, ) # do pca for the style vector data distribution
        var = pca.explained_variance_ # get variance along each pc axis ranked from high to low
        pc = pca.components_ # get the pc ranked from high var to low var
    return var, pc


# Get V and U
var, pc = PCA_test(n_mean_latent)
var_64 = torch.tensor(var[critical_point-shift:512-shift], dtype=torch.float32, device = device)  # [64,]
var_64 = var_64.view(-1, 1)  # [64, 1]
sigma_64 = torch.sqrt(var_64)
v_cap = torch.tensor(pc[critical_point-shift:512-shift,:], dtype=torch.float32, device=device)  # low var pc [64x512] dtype: Tensor
u_cap = torch.tensor(pc[0:critical_point, :], dtype=torch.float32, device=device)  # high var pc [448x512] dtype: Tensor

def get_target_image(sigma_64,v_cap,len_key):
    noise_sample = torch.randn(1, 512, device=device)  # get a bunch of Z
    latent_out = g_ema.style(noise_sample)  # get style vector W from Z
    key = torch.randint(2, (len_key, 1), device=device)  # Get random key
    sk_real = torch.multiply(sigma_64, key)
    new_latent = latent_out + sd_moved * torch.matmul(torch.transpose(sk_real, 0, 1), v_cap)
    imgs, _ = g_ema(
        [new_latent], input_is_latent=True)
    latent_out.detach()
    new_latent.detach()
    return key, imgs, latent_out, new_latent


key, imgs, latent_out, new_latent = get_target_image(sigma_64, v_cap, len_key) # imgs: I
w0 = latent_out
wx = new_latent
latent_in = w0
# get img gen using w0
img_x, _ = g_ema([wx], input_is_latent=True) # G(wx)
img_0, _ = g_ema([latent_in], input_is_latent=True) # G(w0)
loss_phi = F.mse_loss(img_x, imgs)
# Define mse loss
loss = F.mse_loss(img_0, imgs)
# percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=device.startswith("cuda"))
# loss = percept(img_0, imgs).sum()
# Calculate dphi/dw
grad_phi = torch.autograd.grad(loss,w0,create_graph=True)
grad_phi = torch.reshape(grad_phi[0], (512, 1))
beta = torch.matmul(v_cap, grad_phi) #v_cap: [64x512] grad_phi: [512x1]
# grad_phi2 = []
# for i in range(512):
#     grad_phi_i = grad_phi[i]
#     grad_phi2_i = torch.autograd.grad(grad_phi_i, w0, retain_graph=True)
#     grad_phi2.append(grad_phi2_i[0])
# grad_phi2 = torch.stack(grad_phi2, 1)[0].to(device)
print(beta)
l1_crit = torch.nn.L1Loss(size_average=False)
# Reconstruction
relu = torch.nn.ReLU()
sigmoid = torch.nn.Sigmoid()
# get estimated key k
pbar = tqdm(range(step))
for i in pbar:
    beta_s = torch.multiply(beta, sigma_64)
    estimated_key = sigmoid(k)
    sk = sd_moved * torch.multiply(sigma_64, estimated_key)  # [64x1]
    vsk = torch.matmul(torch.transpose(sk, 0, 1), v_cap)  # [1x64] * [64x512]
    new_latent = vsk + w0  # [1 x 512]
    img_approx, _ = g_ema([new_latent], input_is_latent=True)  # G(w0)
    loss = F.mse_loss(img_approx, imgs)
    second_term = sd_moved * torch.matmul(torch.transpose(beta_s, 0, 1),sigmoid(k))
    third_term = False
    if third_term:
        sk = sd_moved * torch.multiply(sigma_64, sigmoid(k))
        vsk = torch.matmul(torch.transpose(v_cap, 0, 1), sk)
        beta2 = torch.matmul(torch.transpose(vsk, 0, 1), grad_phi2)
        beta2 = torch.matmul(beta2, vsk)
    else:
        beta2 = 0
    approx_loss_1st = loss#+second_term+1/2*beta2+loss/32*l1_crit(sigmoid(k),torch.zeros_like(k))
    optimizer.zero_grad()
    g_ema.zero_grad()
    approx_loss_1st.backward(retain_graph=True)
    optimizer.step()
    print(approx_loss_1st.item())
# approx_loss_2nd = approx_loss_1st + 1/2*beta2
estimated_key = torch.round(sigmoid(k))
print(estimated_key)
print(key)
# print("loss by 2nd order taylor expansion:", approx_loss_2nd.item())
key_acc = torch.sum(estimated_key == key)
acc = key_acc/(style_space_dim - critical_point)
sk = sd_moved * torch.multiply(sigma_64, estimated_key)  # [64x1]
vsk = torch.matmul(torch.transpose(sk,0,1), v_cap)  # [1x64] * [64x512]
new_latent = vsk + w0  # [1 x 512]
latent_in = new_latent
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(wx, w0)
img_gen, _ = g_ema([latent_in], input_is_latent=True)  # reconstruct image
loss = F.mse_loss(img_gen, imgs)

a = torch.dot(v_cap[0], v_cap[5])
print(a)
print("loss by 1st order taylor expansion:", approx_loss_1st.item())
print("loss of wx:", loss_phi.item())
print('cosine similarity of (w0, wx):{}, \ncosine similarity of (wx, w_approx): {},  \n'
      'cosine similarity of (w0, w_approx): {}.'.format(cos(wx, w0),cos(wx, new_latent), cos(new_latent, w0)))
print("Reconstruction loss:", loss.item())
print('key accuracy: ', acc)




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

reconstructed = make_image(img_gen)
original_image = make_image(imgs)

result_file = {}
for i, input_name in enumerate(files):

    result_file[input_name] = {
        "img": img_gen[i],
        "latent": latent_in[i]
    }
    img_name = "./projection_Test/"+trials+os.path.splitext(os.path.basename(input_name))[0] + "-reconstructed.png"
    pil_img = Image.fromarray(reconstructed[i])
    pil_img.save(img_name)
    img_name = "./projection_Test/"+trials+os.path.splitext(os.path.basename(input_name))[0] + "-Target.png"
    pil_img = Image.fromarray(original_image[i])
    pil_img.save(img_name)
torch.save(result_file, filename)

