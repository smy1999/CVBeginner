import numpy as np
import cv2
from matplotlib import pyplot as plt

"""
形态学操作根据图像形状进行操作, 一般用于二值化图像
"""

img = cv2.imread('yue_reverse.png', 0)

# 1.腐蚀: 使用卷积核腐蚀, 若卷积核对应原图像像素全为1,则保持原像素; 反之置为0
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)

titles = ['Original Image', 'Erosion']
images = [img, erosion]

plt.subplot()
