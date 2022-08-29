from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import os

class Color_code(object):
    def __init__(self,args):
        self.sigma = args.sigma
        self.num_pixel = 256
        self.channel = 3
        self.watermark_pos_total = np.zeros((self.num_pixel*self.num_pixel,self.channel))
        self.watermark_neg_total = np.zeros((self.num_pixel*self.num_pixel,self.channel))
        self.watermark_total = np.zeros((self.num_pixel*self.num_pixel,self.channel))
        self.org_total = np.zeros((self.num_pixel*self.num_pixel,self.channel))
        self.red = np.array([255,0,0])
        # self.yellow = np.array([255,255,0])
        self.green = np.array([0,255,0])
        self.blue = np.array([0,0,255])
        self.heatmap = np.zeros((256 * 256, 3))
        total_key = [[0,1,2],[0,1],[0,2],[1,2],[0],[1],[2]]
        assert args.key<=len(total_key) and args.key>=0, f"key out of range: {args.key}"
        self.keys = total_key[args.key]
        self.amplifier = args.amplifier
        self.threshold = args.threshold
        self.shift = args.shift
        # self.sample = args.sample

    def rgb2gray(self,rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])



    def get_color_coded_image(self):
        for key in self.keys:
            shift = 508 + key
            if key == 0:
                color = self.red
            # elif key == 1:
            #     color = yellow
            elif key == 1:
                color = self.green
            elif key == 2:
                color = self.blue
            else:
                raise AssertionError('key out of the list')
            negative_path = './visualization/fixed_sigma_{}/shift_{}/watermark_neg_gray/000000.png'.format(self.sigma, shift)
            positive_path = './visualization/fixed_sigma_{}/shift_{}/watermark_pos_gray/000000.png'.format(self.sigma, shift)
            negative = Image.open(negative_path)
            neg = np.array(negative.getdata())
            positive = Image.open(positive_path)
            pos = np.array(positive.getdata())
            watermark = pos-neg
            neg[neg<(max(neg)-self.threshold)] = 0
            pos[pos<(max(pos)-self.threshold)] = 0
            watermark[watermark<(max(watermark)-self.threshold)] = 0
            coeff_neg = self.amplifier/max(neg)
            coeff_pos = self.amplifier/max(pos)
            coeff_watermark = self.amplifier/max(watermark)
            pos = pos.reshape([-1,1]).repeat(3,axis=1)
            neg = neg.reshape([-1,1]).repeat(3,axis=1)
            pos = np.uint8(coeff_pos*pos*color)
            neg = np.uint8(coeff_neg*neg*color)
            watermark = watermark.reshape([-1,1]).repeat(3,axis=1)
            watermark = coeff_watermark*watermark*color

            watermark_pos_total = np.add(self.watermark_pos_total,pos)
            watermark_neg_total = np.add(self.watermark_neg_total,neg)
            watermark_total = np.add(self.watermark_total,watermark)

        watermark_neg_total = self.watermark_neg_total.clip(0,255)/(255)
        watermark_pos_total = self.watermark_pos_total.clip(0,255)/(255)
        watermark_total = self.watermark_total.clip(0,255)/(255)
        original_image_path = './result_images/fixed_sigma_{}/shift_{}/watermark_pos/000000.png'.format(self.sigma, self.shift)
        original = Image.open(original_image_path)
        org = np.array(original.getdata())
        org_total = np.add(self.org_total,org)
        # plt.imshow(watermark_pos_total.reshape((256,256,3)))
        # plt.show()
        #
        # plt.imshow(watermark_neg_total.reshape((256,256,3)))
        # plt.show()
        plt.imshow(watermark_total.reshape((256,256,3)))
        plt.show()

    def get_heat_map(self,sample):
        negative_path = './visualization/fixed_sigma_{}/shift_{}/watermark_neg/'.format(self.sigma, self.shift) + f'{sample:06d}.png'
        positive_path = './visualization/fixed_sigma_{}/shift_{}/watermark_pos/'.format(self.sigma, self.shift) + f'{sample:06d}.png'
        negative = Image.open(negative_path)
        neg = np.array(negative.getdata())
        neg = self.rgb2gray(neg)
        positive = Image.open(positive_path)
        pos = np.array(positive.getdata())
        pos = self.rgb2gray(pos)

        for iter, element in enumerate(pos):
            self.heatmap[iter][0] = element
        for iter, element in enumerate(neg):
            self.heatmap[iter][2] = element
        heatmap = self.amplifier * self.heatmap / 256
        plt.imshow(heatmap.reshape((self.num_pixel, self.num_pixel, self.channel)))
        # plt.show()
        plt.axis('off')
        plt.savefig('./visualization/sigma_{}_shift_{}_sample_{}.png'.format(self.sigma,self.shift,sample), bbox_inches='tight', pad_inches=0)

    def get_superimposing_heatmap(self,sample):
        negative_path = './visualization/fixed_sigma_{}/shift_{}/watermark_neg/'.format(self.sigma, self.shift) + f'{sample:06d}.png'
        positive_path = './visualization/fixed_sigma_{}/shift_{}/watermark_pos/'.format(self.sigma, self.shift) + f'{sample:06d}.png'
        original_image_path = './visualization/fixed_sigma_{}/shift_{}/target_perturbed/'.format(self.sigma, self.shift) + f'{sample:06d}.png'
        neg = cv2.imread(negative_path, cv2.IMREAD_GRAYSCALE)
        pos = cv2.imread(positive_path, cv2.IMREAD_GRAYSCALE)
        org = cv2.imread(original_image_path)
        # neg = np.array(negative.getdata())
        # neg = self.rgb2gray(neg)
        #
        # pos = np.array(positive.getdata())
        # pos = self.rgb2gray(pos)

        image = np.uint8(pos+neg)
        th = cv2.threshold(image, 25, 255, cv2.THRESH_BINARY_INV)[1]
        th = th.reshape((256,256,1))
        blur = cv2.GaussianBlur(th, (5, 5), 5)
        heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
        super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, org, 0.8, 0)


        plt.imshow(super_imposed_img.reshape((self.num_pixel, self.num_pixel, 3)))
        # plt.imshow(th.reshape((self.num_pixel, self.num_pixel, 1)))
        plt.axis('off')
        path = './visualization/sigma_{}/shift_{}/'.format(self.sigma,self.shift)
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        plt.savefig(path + 'sample_{}.png'.format(sample), bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image generator for generating perturbed images"
    )
    parser.add_argument(
        "--key", type=int, default=6, help="range from 0 to 6"
    )
    parser.add_argument(
        "--amplifier", type=float, default=10, help="watermark amplifier"
    )
    parser.add_argument(
        "--threshold", type=int, default=100, help="remove pixels smaller than some threshold"
    )
    parser.add_argument(
        "--shift", type=int, default=16, help="starting index of editing direction"
    )
    parser.add_argument(
        "--sigma", type=int, default=1, help="editing direction"
    )
    # parser.add_argument(
    #     "--sample", type=int, default=8, help="editing direction"
    # )



    args = parser.parse_args()
    color_code = Color_code(args)
    for sample in range(100):
        color_code.get_superimposing_heatmap(sample)




