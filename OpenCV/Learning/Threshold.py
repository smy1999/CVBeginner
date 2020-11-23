import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../src/image.jpg', 0)

# param : 图像/阈值/置于的量/方式
# return: 阈值/新图像
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

title = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray'), plt.title(title[i])
    plt.xticks([]), plt.yticks([])
# plt.show()

img = cv2.medianBlur(img, 5)  # 中值滤波
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# 图像/目标值/方法/邻域大小/常数(阈值等于平均或加权平均减该常数)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# cv2.ADAPTIVE_THRESH_MEAN_C        阈值取自邻域平均值
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C    阈值取自邻域加权平均值, 权重大小为一个高斯窗口

title = ['Original Image', 'Global Thresholding v=127', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
image = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray'), plt.title(title[i])
    plt.xticks([]), plt.yticks([])
plt.show()
"""
cv2.THRESH_BINARY       大于阈值置255,小于置0
cv2.THRESH_BINARY_INV   大于阈值置0,小于置255
cv2.THRESH_TRUNC        大于阈值置阈值,小于不变
cv2.THRESH_TOZERO       小于阈值置0,大于不变
cv2.THRESH_TOZERO_INV   大于阈值置0,小于不变
"""