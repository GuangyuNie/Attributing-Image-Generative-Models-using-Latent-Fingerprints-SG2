## Deep Watermarks For Attributing Generative Models

![Teaser image](./image/iclr2023_fig1.png)
Abstract: Generative models have enabled the creation of contents that are indistinguishable from those taken from the Nature. 
Open-source development of such models raised concerns about the risks in their misuse for malicious purposes.
One potential risk mitigation strategy is to attribute generative models via watermarking. 
Current watermarking methods exhibit significant tradeoff between robust attribution accuracy and generation quality, 
and also lack principles for designing watermarks to improve this tradeoff. 
This paper investigates the use of latent semantic dimensions as watermarks, from where we can analyze the effects of design variables, 
including the choice of watermarking dimensions, watermarking strength, and the capacity of watermarks, on the 
accuracy-quality tradeoff.Compared with previous SOTA, our method requires minimum compute and is more applicable to 
large-scale models. We use StyleGAN2 and the latent diffusion model to demonstrate the efficacy of our method.

## Prerequisites

- NVIDIA GPU + CUDA 10.1 or above
- Python 3.7 or above
- pytorch 1.11 or above
- Anaconda recommended
- To install the other Python dependencies using anaconda, run `conda env create -f env.yml`.

## Checkpoints

We experiment on FFHQ, AFHQ-cat, and AFHQ-dog. 
Checkpoints can be downloaded below.\
Make checkpoint folder in src folder and put weights under src/checkpoint/.\
Pretrained Weights:\
[FFHQ](https://github.com/rosinality/stylegan2-pytorch)\
[AFHQ](https://github.com/NVlabs/stylegan2-ada)

## Generate watermarked image

- Run, e.g.,
  ```
  python generator.py --model sg2 --save_dir '../result/' --key_len 64 --sigma 1 --shift 448
  ```
  where
- `save_dir`: Directory for output saving
- `key_len`: Digits of binary key, higher key length will increase the key capacity. For key length = 64, the key capacity would be 2^64
- `sigma` : Watermarking perturbation strength
- `shift` : Initial index of consecutive principal axis ranked from a descent order based on it's corresponding variance. 
E.g. the set of editing direction V follows V = PC[shift:shift+key_len]  
After running the code, watermarked images will be saved under result folder. 

## Attribution of watermarked image

- Run, e.g.,
  ```
  python main.py --model sg2 --save_dir '../result/' --key_len 64 --sigma 1 --shift 448 --step 2000 --sample_size 100 --n 20
  ```
  besides the argumentation from above, we have additional argument:
- `step`: optimization steps to attribute watermarked image
- `sample_size`: number of attribution tests user would like to perform
- `n`: Number of initial guess from Latin Hypercube sampling method

The result will be saved under result folder.



