import cv2
import numpy as np
import os

# assert os.path.exists('./result_images/fixed_sigma_1/shift_448/target_perturbed/target_wx_0.png')
a = []
for i in range(100):
    # img1 = cv2.imread('./result_images/fixed_sigma_06/shift_448/target_perturbed/target_wx_{}.png'.format(i))
    # img2 = cv2.imread('./result_images/fixed_sigma_06/shift_448/target_before_perturb/target_w0_{}.png'.format(i))
    img1 = cv2.imread('./test_images/key_128_sd_4_aug_None/image_before_perturb/target_w0_{}.png'.format(i))
    img2 = cv2.imread('./test_images/key_128_sd_4_aug_None/perturbed_image/target_wx_{}.png'.format(i))
    psnr = cv2.PSNR(img1, img2)
    a.append(psnr)
a = np.array(a)
print(np.std(a))
print(np.mean(a))