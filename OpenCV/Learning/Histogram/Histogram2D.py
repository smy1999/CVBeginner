import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
彩色图像直方图考虑颜色Hue和饱和度Saturation
二维BGR2HSV 一维BGR2GRAY
"""

img = cv2.imread('../../src/desktop.jpg')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# cv2绘制2d直方图
# 图像/01通道/掩模/H180S256/H0-180S0-256
hist_cv2 = cv2.calcHist([img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

# numpy绘制2d直方图
# H通道/S通道/bin数目/取值范围
hist_np, xbins, ybins = np.histogram2d(img_hsv[0].ravel(), img_hsv[1].ravel(), [180, 256], [[0, 180], [0, 256]])

# 使用cv2显示直方图
cv2.imshow('Histogram Opencv', hist_cv2)
cv2.imshow('Histogram Numpy', hist_np)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 使用Matplotlib显示直方图(better)
plt.imshow(hist_cv2, interpolation='nearest')  # 须为nearest
plt.show()
