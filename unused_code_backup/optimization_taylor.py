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
# files = ['./projection_Test/000000.png',
#          './projection_Test/000001.png',
#          './projection_Test/000002.png',
#          './projection_Test/000003.png',
#          './projection_Test/000004.png']
#files = ['./projection_Test/000000.png']
files = ['./sample_shifted_448/000018.png']
n_mean_latent = 10000 # num of style vector to sample
img_size = 256 # image size
style_space_dim = 512
mapping_network_layer = 8
resize = min(img_size, 256)
critical_point = 448 # 512 - 64, num of high var pc
init_lr = 0.1 # learning rate
step = 1000
use_mse_loss = True

def get_lr(i,step,init_lr, decay_rate=1e-4, cap=0.01, decay_start=0.5):
    """learning rate decay"""
    if i/step > decay_start:
        lr = 1/(1+decay_rate*(i-decay_start*step))*init_lr
        lr = min(lr, cap)
    else:
        lr = init_lr
    return lr

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

# Do PCA
pca = PCA()
with torch.no_grad():
    noise_sample = torch.randn(n_mean_latent, 512, device=device) # get a bunch of Z
    latent_out = g_ema.style(noise_sample) # get style vector from Z
    latent_mean = latent_out.mean(0) # take an avg of all style vector, let this be our init guess
    latent_out = latent_out.detach().cpu().numpy()
    latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
    pca.fit(latent_out) # do pca for the style vector data distribution
    var = pca.explained_variance_ # get variance along each pc axis ranked from high to low
    pc = pca.components_ # get the pc ranked from high var to low var

# Get V and U
var_64 = torch.tensor(var[critical_point:512], device = device)
var_64 = var_64.view(-1,1)
sigma_64 = torch.sqrt(var_64)
v_cap = torch.tensor(pc[critical_point:512,:], device=device) # low var pc [64x512] dtype: Tensor
u_cap = torch.tensor(pc[0:critical_point, :], device=device)# high var pc [448x512] dtype: Tensor
v_cap_t = torch.transpose(v_cap, 0, 1) # [512x64] tensor
u_cap_t = torch.transpose(u_cap, 0, 1) # [512x448] tensor
percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=device.startswith("cuda")) # define perceptual loss
latent_mean = torch.reshape(latent_mean, (512, 1))

# get init guess for alpha
alpha, _ = torch.lstsq(latent_mean,u_cap_t) # solve for init of for alpha = [512x1] tensor
init_alpha = alpha[0:critical_point,:] # solution for alpha = [448 x 1] tensor

# Define optimizer
init_alpha.requires_grad = True
optimizer = optim.Adam([init_alpha], lr=init_lr)

# optimize Iteration
pbar = tqdm(range(step))
latent_path = []
loss1 = []
loss2 = []
lossf = []
relu = torch.nn.ReLU()
for i in pbar:
    latent = torch.matmul(u_cap_t, init_alpha)  # get [512 x 1] style vector
    latent.retain_grad()
    latent_re = torch.reshape(latent, (1, 512))  # [1 x 512]
    latent_in = latent_re.clone().repeat(imgs.shape[0], 1)  # [batch x 512]
    latent_in_re = torch.reshape(latent_in, (imgs.shape[0], 512))
    # lr = get_lr(i,step, init_lr)
    # optimizer.param_groups[0]["lr"] = lr
    img_gen, _ = g_ema([latent_in], input_is_latent=True)
    img_gen.retain_grad()
    batch, channel, height, width = img_gen.shape

    if height > 256:
        factor = height // 256

        img_gen = img_gen.reshape(
            batch, channel, height // factor, factor, width // factor, factor
        )
        img_gen = img_gen.mean([3, 5])

    p_loss = percept(img_gen, imgs).sum()
    mse_loss = F.mse_loss(img_gen, imgs)
    mse_latent = F.mse_loss(init_alpha, alpha[0:critical_point,:])
    if use_mse_loss:
        loss = mse_loss
    else:
        loss = p_loss
    grad_phi = torch.autograd.grad(loss,latent,create_graph=True)
    beta = torch.matmul(v_cap, grad_phi[0]) #v_cap: [64x512] grad_phi: [512x1]
    loss_1st = torch.multiply(sigma_64, -72*relu(-1*beta))  # if positive return 0, if negative return value
    # if 1:
    #     k = 72*relu(-1*torch.sign(beta))  # get the key
    #     sk = torch.multiply(sigma_64, k)
    #     vsk = torch.matmul(v_cap_t, sk)
    #     new_latent = vsk + latent
    #     new_latent = torch.reshape(new_latent, (1, 512))  # [1 x 512]
    #     new_latent = new_latent.clone().repeat(imgs.shape[0], 1)  # [batch x 512]
    #     img_gen_recon, _ = g_ema([new_latent], input_is_latent=True, have_noise = False)
    #     recon_lost = F.mse_loss(img_gen_recon, imgs)
    #     loss_1st = torch.matmul(torch.transpose(vsk,0,1), grad_phi[0])
    #     final_loss = abs(recon_lost - (loss + loss_1st)) + recon_lost
    final_loss = loss + torch.sum(loss_1st)
    optimizer.zero_grad()
    g_ema.zero_grad()
    final_loss.backward()
    optimizer.step()
    if (i + 1) % 100 == 0:
        latent_path.append(latent_in.detach().clone())
    loss1.append(mse_loss.item())
    loss2.append(torch.sum(loss_1st).item())
    lossf.append(final_loss.item())
    pbar.set_description(
        (
            f"perceptual: {p_loss.item():.4f}"
            f" mse: {mse_loss.item():.4f};"
            f" loss_1st: {torch.sum(loss_1st):.4f};"
            #f" recon_lost: {recon_lost:.4f};"

        )
    )
print(beta)



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

img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True)

trials = 'trial_1_original_1/'
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

# plt.figure()
# plt.plot(loss1)
# plt.plot(loss2)
# plt.plot(lossf)
# plt.ylim([-3, 3])
# plt.xlabel('training iteration')
# plt.ylabel('loss')
# plt.legend(['loss for the first term', 'loss for the second term', 'total loss'])
# plt.savefig(test_output_dir + 'loss1.png')
# plt.show()
#
# plt.figure()
# plt.plot(loss1)
# plt.plot(loss2)
# plt.plot(lossf)
# plt.xlabel('training iteration')
# plt.ylabel('loss')
# plt.legend(['loss for the first term', 'loss for the second term', 'total loss'])
# plt.savefig(test_output_dir + 'loss2.png')
# plt.show()
