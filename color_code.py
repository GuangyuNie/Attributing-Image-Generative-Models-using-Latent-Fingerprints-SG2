from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2



sigma = '3'
watermark_pos_total = np.zeros((256*256,3))
watermark_neg_total = np.zeros((256*256,3))
watermark_total = np.zeros((256*256,3))
org_total = np.zeros((256*256*1,3))

red = np.array([255,0,0])
# yellow = np.array([255,255,0])
green = np.array([0,255,0])
blue = np.array([0,0,255])


keys = [[0,1,2],[0,1],[0,2],[1,2],[0],[1],[2]]
keys = keys[6]
amplifier = 2
threshold = 100
for key in keys:
    shift = 508 + key
    if key == 0:
        color = red
    # elif key == 1:
    #     color = yellow
    elif key == 1:
        color = green
    elif key == 2:
        color = blue
    negative_path = './visualization/fixed_sigma_{}/shift_{}/watermark_neg_gray/watermark_0.png'.format(sigma, shift)
    positive_path = './visualization/fixed_sigma_{}/shift_{}/watermark_pos_gray/watermark_0.png'.format(sigma, shift)
    negative = Image.open(negative_path)
    neg = np.array(negative.getdata())
    positive = Image.open(positive_path)
    pos = np.array(positive.getdata())
    watermark = pos-neg
    neg[neg<(max(neg)-threshold)] = 0
    pos[pos<(max(pos)-threshold)] = 0
    watermark[watermark<(max(watermark)-threshold)] = 0
    coeff_neg = amplifier/max(neg)
    coeff_pos = amplifier/max(pos)
    coeff_watermark = amplifier/max(watermark)
    pos = pos.reshape([-1,1]).repeat(3,axis=1)
    neg = neg.reshape([-1,1]).repeat(3,axis=1)
    pos = np.uint8(coeff_pos*pos*color)
    neg = np.uint8(coeff_neg*neg*color)
    watermark = watermark.reshape([-1,1]).repeat(3,axis=1)
    watermark = coeff_watermark*watermark*color

    watermark_pos_total = np.add(watermark_pos_total,pos)
    watermark_neg_total = np.add(watermark_neg_total,neg)
    watermark_total = np.add(watermark_total,watermark)

watermark_neg_total = watermark_neg_total.clip(0,255)/(255)
watermark_pos_total = watermark_pos_total.clip(0,255)/(255)
watermark_total = watermark_total.clip(0,255)/(255)
original_image_path = './result_images/fixed_sigma_{}/shift_{}/watermark_pos/watermark_0.png'.format(sigma, 0)
original = Image.open(original_image_path)
org = np.array(original.getdata())
org_total = np.add(org_total,org)
# plt.imshow(watermark_pos_total.reshape((256,256,3)))
# plt.show()
#
# plt.imshow(watermark_neg_total.reshape((256,256,3)))
# plt.show()
plt.imshow(watermark_total.reshape((256,256,3)))
plt.show()

#
# heatmap = np.zeros((256*256,3))
# for iter, element in enumerate(watermark_pos_total):
#     heatmap[iter][0] = element
# for iter, element in enumerate(watermark_neg_total):
#     heatmap[iter][2] = element
#
# heatmap = 2*heatmap/256
# plt.imshow(heatmap.reshape((256,256,3)))
# plt.show()
#
# data = Image.fromarray(np.reshape(watermark_pos_total,(256,256)))
# plt.imshow(data)
# plt.show()
#
# data = Image.fromarray(np.reshape(watermark_neg_total,(256,256)))
# plt.imshow(data)
# plt.show()
#
#
# data = Image.fromarray(np.reshape(org_total-watermark_neg_total+watermark_pos_total,(256,256,1)))
# plt.imshow(data)
# plt.show()
# shift = 0
# sigma = 3
# negative_path = './result_images/fixed_sigma_{}/shift_{}/watermark_neg/watermark_0.png'.format(sigma, shift)
# positive_path = './result_images/fixed_sigma_{}/shift_{}/watermark_pos/watermark_0.png'.format(sigma, shift)
# negative = Image.open(negative_path)
# neg = np.array(negative.getdata())
# positive = Image.open(positive_path)
# pos = np.array(positive.getdata())
#
# heatmap = np.zeros((256*256,3))
# for iter, element in enumerate(pos):
#     heatmap[iter][0] = element
# for iter, element in enumerate(neg):
#     heatmap[iter][2] = element
# heatmap = 2*heatmap/256
# plt.imshow(heatmap.reshape((256,256,3)))
# plt.show()




