import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
cv2快于numpy
BINS : 横轴坐标区间数
DIMS : 收集的参数数目
RANGE: 灰度值范围
"""

img = cv2.imread('../../src/desktop.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 图像/通道(灰度为0,彩色0B1G2R/掩模/BIN数目/RANGE像素范围,除掩模其余均用[],返回256*1的数组,对应与灰度值对应的像素点数目
hist1 = cv2.calcHist([img], [0], None, [256], [0, 256])
# 图像(转一维数组)/BIN/RANGE,返回值为257,是由于最后范围是255-255.99,外加256
hist2, bins = np.histogram(img.ravel(), 256, [0, 256])
# 图像/ ,速度快,适合一维直方图
hist3 = np.bincount(img.ravel(), minlength=256)

# 使用Matplotlib绘制直方图
plt.hist(img_gray.ravel(), 256, [0, 256])
plt.show()

# 使用Matplotlib绘制多通道直方图
colors = ('b', 'g', 'r')
for i, color in enumerate(colors):  # enumerate用于同时遍历索引和元素
    hist_rgb = cv2.calcHist(img, [i], None, [256], [0, 256])
    plt.plot(hist_rgb, color=color)
    plt.xlim([0, 256])
plt.show()

# 使用Matplotlib绘制带掩模的直方图, 目标区域白, 非目标区域黑
mask = np.zeros(img.shape[:2], np.uint8)
mask[200:600, 400:1000] = 255
img_mask = cv2.bitwise_and(img_gray, img_gray, mask=mask)
hist_full = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img_gray], [0], mask, [256], [0, 256])

plt.subplot(221), plt.imshow(img_gray, 'gray'), plt.title('Original Image')
plt.subplot(222), plt.imshow(mask, 'gray'), plt.title('Mask')
plt.subplot(223), plt.imshow(img_mask, 'gray'), plt.title('Masked Image')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask), plt.xlim([0, 256])
plt.show()
