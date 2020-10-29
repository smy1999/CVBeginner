import numpy as np
import cv2
from matplotlib import pyplot as plt

"""
形态学操作根据图像形状进行操作, 一般用于二值化图像
"""

img = cv2.imread('yue_reverse.png', 0)

kernel = np.ones((5, 5), np.uint8)  # 构建5*5的卷积核

# 1.腐蚀: 使用卷积核腐蚀, 若卷积核对应原图像像素全为1,则保持原像素; 反之置为0
erosion = cv2.erode(img, kernel, iterations=1)

# 2.膨胀: 若卷积核对应图像的像素有一个是1, 就是1
dilation = cv2.dilate(img, kernel, iterations=1)

# 3.开运算: 先腐蚀, 再膨胀. 用于去除噪声
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# 4.闭运算: 先膨胀, 再腐蚀. 用于填充前景物体的小洞
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# 5.形态学梯度: 膨胀与腐蚀的差, 用于查看前景物体的轮廓
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# 6.礼帽: 原始图像与开运算得到的差. 得到噪声信息
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

# 7.黑帽: 闭运算与原始图像的差. 突出显示原图亮区域周围的暗区域
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

titles = ['Original Image', 'Erosion', 'Dilation', 'Opening', 'Closing', 'Gradient', 'Top Hat', 'Black Hat']
images = [img, erosion, dilation, opening, closing, gradient, tophat, blackhat]

for i in range(8):
    plt.subplot(3, 3, i + 1), plt.imshow(images[i], cmap='gray'), plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# 创建核函数
# Rectangular Kernel
rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# Elliptical Kernel
ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# Cross-shaped Kernel
cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
