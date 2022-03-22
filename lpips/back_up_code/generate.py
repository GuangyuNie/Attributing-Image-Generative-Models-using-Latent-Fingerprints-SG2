import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
from sklearn.decomposition import PCA
import math
import numpy as np
import os


def generate(args, g_ema, device, mean_latent):
    torch.manual_seed(2022)
    data_sample = []
    total_data_sampled = 10000
    get_image = True
    do_PCA = True
    get_shifted_image = True
    with torch.no_grad():
        g_ema.eval()
        if get_image:
            for i in tqdm(range(args.pics)):
                sample_z = torch.randn(args.sample, args.latent, device=device)

                sample, style_w = g_ema(
                    [sample_z], truncation=args.truncation, truncation_latent=mean_latent
                )
                utils.save_image(
                    sample,
                    f"sample/{str(i).zfill(6)}.png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
        if do_PCA:
            for i in range(total_data_sampled):
                sample_z = torch.randn(args.sample, args.latent, device=device)

                style_w = g_ema(
                    [sample_z], get_latent_only=True, truncation=args.truncation, truncation_latent=mean_latent
                )
                data_sample.append(style_w[0][0].detach().cpu().numpy())
            pca = PCA()
            pca.fit(data_sample)
            eig_val = pca.explained_variance_
            eig_vec = pca.components_
            np.save('data_sample.npy', data_sample)
            np.save('eig_val.npy', eig_val)
            np.save('eig_vec.npy', eig_vec)

        if get_shifted_image:
            eig_testing = 0
            test_output_dir = './sample_shifted_{}'.format(eig_testing)
            isExist = os.path.exists(test_output_dir)
            if not isExist:
                os.makedirs(test_output_dir)
            eig_val = np.load('eig_val.npy')
            eig_vec = np.load('eig_vec.npy')
            data_sample = np.load('data_sample.npy')
            for i in tqdm(range(args.pics)):
                new_latent = [data_sample[i]+0*math.sqrt(eig_val[eig_testing])*eig_vec[eig_testing]]
                new_latent = torch.tensor(new_latent, dtype=torch.float32, device=device)

                sample_shifted, _ = g_ema(
                    [new_latent], input_is_latent=True)
                utils.save_image(
                    sample_shifted,
                    test_output_dir+f"/{str(i).zfill(6)}.png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./checkpoint/550000.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"],strict=False)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
