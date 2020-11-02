import cv2
import numpy as np
from matplotlib import pyplot as plt
"""
对直方图横向拉伸, 使图像分布区间广, 改善对比度
"""

img = cv2.imread('../../src/abnormalized.jpg', 0)

# flatten()转为一维数组
hist, bins = np.histogram(img.flatten(), 256, [0, 256])
# 计算累积分布图
cdf = hist.cumsum()  # 将之前的值加到本数 ex:(1,2,3)->(1,3,6)
cdf_normalized = cdf * hist.max() / cdf.max()  # 归一化

plt.subplot(222), plt.title('Original Histogram')
plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')  # 图例

# numpy直方图均衡化
cdf_mask = np.ma.masked_equal(cdf, 0)  # 构建numpy掩模数组,cdf数组当元素为0时掩盖(计算时忽略)
cdf_mask = (cdf_mask - cdf_mask.min()) * 255 / (cdf_mask.max() - cdf_mask.min())  # 归一化
cdf = np.ma.filled(cdf_mask, 0).astype('uint8')  # 补0, 取整

img_normalized = cdf[img]  # 将变换应用到图像上

hist, bins = np.histogram(img_normalized.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()
plt.subplot(224), plt.title('Normalized Histogram')
plt.plot(cdf_normalized, color='b')
plt.hist(img_normalized.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')  # 图例

plt.subplot(221), plt.imshow(img, 'gray'), plt.title('Original Image')
plt.subplot(223), plt.imshow(img_normalized, 'gray'), plt.title('Normalized Image')
plt.show()

# 2.cv2的直方图均衡
equ = cv2.equalizeHist(img)  # 直方图均衡
res = np.hstack((img, equ))
cv2.imshow('cv2 Histogram Normalize', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
