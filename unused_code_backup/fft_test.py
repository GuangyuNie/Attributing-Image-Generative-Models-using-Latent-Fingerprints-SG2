from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist
i = 0

image_dir = ['../sample_shifted_0/000000.png', '../sample_shifted_100/000000.png','../sample_shifted_500/000000.png'
             ,'../sample_shifted_0/000005.png','../sample_shifted_100/000005.png','../sample_shifted_500/000005.png'
             ,'../sample_shifted_0/000010.png','../sample_shifted_100/000010.png','../sample_shifted_500/000010.png']
save_dir = ['../frequency_domain/freq_0_0', '../frequency_domain/freq_100_0', '../frequency_domain/freq_500_0',
            '../frequency_domain/freq_0_5', '../frequency_domain/freq_100_5', '../frequency_domain/freq_500_5',
            '../frequency_domain/freq_0_10','../frequency_domain/freq_100_10','../frequency_domain/freq_500_10']
for i in range(len(image_dir)):
    img = plt.imread(image_dir[i]).astype(float)
    img_r = img[0]
    img_g = img[1]
    img_b = img[2]


    dark_image_grey = rgb2gray(img)
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(dark_image_grey, cmap='gray');

    dark_image_grey_fourier = np.fft.fft2(dark_image_grey)
    shifted = np.fft.fftshift(dark_image_grey_fourier)
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(np.log(abs(shifted)), cmap='gray');
    plt.savefig(save_dir[i])
    # plt.show()

    ifft_image = np.abs(np.fft.ifft2(dark_image_grey_fourier))
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(ifft_image, plt.cm.gray, vmin=0, vmax=1);
    # plt.show()