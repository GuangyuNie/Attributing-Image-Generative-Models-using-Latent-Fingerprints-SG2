from skimage.metrics import peak_signal_noise_ratio
from PIL import Image
import numpy as np
import os

class PSNR():
    def __init__(self, datarange):
        self.datarange = datarange #need maximum difference between min and max value of img. For png, 256.

    def cal_psnr(self, img1, img2):
        return peak_signal_noise_ratio(img1, img2, data_range=self.datarange)




if __name__ == "__main__":
    psnr = PSNR(256)

    folder1 = "./test_images/image_before_perturb/"
    folder2 = "./test_images/perturbed_image/"

    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)

    assert len(files1) == len(files2)

    result = 0

    for i in range(len(files1)):
        b4_template = "target_w0_" + str(i) + ".png"
        pertrubed_template = "target_wx_"+str(i)+".png"

        img1 = np.asarray(Image.open(folder1 + b4_template))
        img2 = np.asarray(Image.open(folder2 + pertrubed_template))
        #print(psnr.cal_psnr(img1, img2))

        result += psnr.cal_psnr(img1,img2)

    print("Averaged PNSR: {}".format(str(result / len(files1))))