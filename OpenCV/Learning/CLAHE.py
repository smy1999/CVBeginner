import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
CLAHE : Contrast Limited Adaptive Histogram Equalization 限制对比度自适应直方图均衡
将图像分成小块, 对每个小块分别进行直方图均衡化, 但在每块中噪声会放大, 故对比度限制
若直方图中的bin超过对比度上限, 就将像素均匀分散到其他的bin, 再直方图均衡化
最后使用双线性插值去除小块的边界
"""

img = cv2.imread('../src/clahe.png', 0)

# 对比度阈值/像素块大小
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img)

img_he = cv2.equalizeHist(img)

plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
plt.subplot(132), plt.imshow(img_he, 'gray'), plt.title('Global Histogram Equalization')
plt.subplot(133), plt.imshow(img_clahe, 'gray'), plt.title('CLAHE')
plt.show()
