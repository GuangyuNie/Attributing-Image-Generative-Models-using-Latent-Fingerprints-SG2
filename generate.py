import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
from sklearn.decomposition import PCA
import math
from PIL import Image
import numpy as np
import os


def generate(args, g_ema, device, mean_latent):
    torch.manual_seed(2022)
    # training dataset seed 2022
    data_sample = []
    total_data_sampled = 10000
    get_image = False
    do_PCA = False
    get_shifted_image = True
    get_key_image = False
    style_mixing = False
    test_output_dir = './data/train/'
    image_data = []
    Phi = []
    with torch.no_grad():
        g_ema.eval()
        if get_image:
            for i in tqdm(range(total_data_sampled)):
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
                sample = sample.detach().cpu().numpy()
                image_data.append(sample)
            np.save('./image_data', image_data)
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
            np.save(test_output_dir+'data_sample.npy', data_sample)
            np.save(test_output_dir+'eig_val.npy', eig_val)
            np.save(test_output_dir+'eig_vec.npy', eig_vec)

        if get_shifted_image:
            #eig_testing = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500]
            eig_testing = [0]

            eig_val = np.load('eig_val.npy')
            eig_vec = np.load('eig_vec.npy')
            data_sample = np.load('data_sample.npy')
            for eig_index in eig_testing:
                test_output_dir = './sample_shifted_{}'.format(eig_index)
                isExist = os.path.exists(test_output_dir)
                if not isExist:
                    os.makedirs(test_output_dir)

                for i in tqdm(range(args.pics)):
                    new_latent = [data_sample[i]+0*math.sqrt(eig_val[eig_index])*eig_vec[eig_index]]
                    Phi.append(new_latent)
                    new_latent = torch.tensor(new_latent, dtype=torch.float32, device=device)
                    data_sample_i = torch.tensor([data_sample[i]], dtype=torch.float32, device=device)
                    if style_mixing:
                        image_shifted, _ = g_ema(
                            [data_sample_i, new_latent], input_is_latent=True, truncation=args.truncation, truncation_latent=mean_latent)
                    else:
                        image_shifted, _ = g_ema(
                            [new_latent], input_is_latent=True, truncation=args.truncation,
                            truncation_latent=mean_latent)
                    utils.save_image(
                        image_shifted,
                        test_output_dir+f"/{str(i).zfill(6)}.png",
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                    )
                    image_np = image_shifted.detach().cpu().numpy()
                    np.save(test_output_dir+'/image_{}.npy'.format(i),image_np)
                np.save(test_output_dir+'/phi.npy',Phi)
        if get_key_image:
            num_feature = 512
            key_len = 64
            num_data = 20000
            starting_pos = num_feature - key_len
            eig_val = np.load('./data_distribution_1M/eig_val.npy')
            eig_vec = np.load('./data_distribution_1M/eig_vec.npy')
            data_sample = np.load('./data_distribution_1M/data_sample.npy')
            isExist = os.path.exists(test_output_dir)
            if not isExist:
                os.makedirs(test_output_dir)
            label = []
            key = []
            num_classes = 10
            for k in range(num_classes):
                rand_code = np.random.randint(2, size=key_len)
                isExist = os.path.exists(test_output_dir+str(k))
                if not isExist:
                    os.makedirs(test_output_dir+str(k))
                for i in tqdm(range(num_data)):
                    new_latent = data_sample[i]
                    for j in range(len(rand_code)):
                        new_latent += rand_code[j]*6*math.sqrt(eig_val[starting_pos+j])*eig_vec[starting_pos+j]
                    label.append([new_latent])
                    key.append([rand_code])
                    new_latent = torch.tensor([new_latent], dtype=torch.float32, device=device)
                    data_sample_i = torch.tensor([data_sample[i]], dtype=torch.float32, device=device)
                    if style_mixing:
                        image_shifted, _ = g_ema(
                            [data_sample_i, new_latent], input_is_latent=True, truncation=args.truncation,
                            truncation_latent=mean_latent)
                    else:
                        image_shifted, _ = g_ema(
                            [new_latent], input_is_latent=True, truncation=args.truncation,
                            truncation_latent=mean_latent)
                    utils.save_image(
                        image_shifted,
                        test_output_dir+str(k)+f"/{str(i).zfill(6)}.png",
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                    )
                label = np.asarray(label)
                np.save(test_output_dir+'label.npy', label)
                np.save(test_output_dir+'key.npy', key)


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
