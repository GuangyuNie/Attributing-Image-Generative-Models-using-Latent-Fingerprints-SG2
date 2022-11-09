import argparse

parser = argparse.ArgumentParser(description="deep watermarks for generative model")
parser.add_argument("--model", type=str, default='sg2', required=True, help="GAN model: sg2 | biggan")
parser.add_argument("--biggan_label", type=str, default='golden retriever', required=False,
                    help="Biggan label to generate image")
parser.add_argument("--ckpt", type=str, default='./checkpoint/550000.pt', required=False,
                    help="path to the model checkpoint")
parser.add_argument("--img_size", type=int, default=256, help="output image sizes of the generator")
parser.add_argument("--sample_size", type=int, default=100, help="Number of sample generated")
parser.add_argument("--sd", type=int, default=1, help="Standard deviation moved")
parser.add_argument("--steps", type=int, default=2000, help="Number of optimization steps")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generating images")
parser.add_argument("--n", type=int, default=20, help="Number of samples for Latin hypercube sampling method")
parser.add_argument("--key_len", type=int, default=64, help="Number of digit for the binary key")
parser.add_argument("--save_dir", type=str, default='../result/', help="Directory for result and image saving")
parser.add_argument("--augmentation", type=str, default='None',
                    help="Augmentation method: Crop, Noise, Blur, Jpeg, Combination ")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU device id")
parser.add_argument("--shift", type=int, default=448, help="initial PC to perturb, e.g. V = [shift:shift+key_len]")
parser.add_argument("--sigma", type=float, default=1.0, help="perturb strength")
parser.add_argument("--lr", type=float, default=0.2, help="perturb strength")

opt = parser.parse_args()
opt.device = 'cuda:{}'.format(opt.gpu_id)

