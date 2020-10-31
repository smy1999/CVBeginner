import cv2
from matplotlib import pyplot as plt

"""
OTSU算法:
在直方图的两个峰中需要阈值t, 将两峰分开, 并且使每一个峰内的方差最小
"""

img = cv2.imread('../src/coin_noise.jpg', 0)

ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# otsu算法二值化, 返回值ret即为阈值
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

blur = cv2.GaussianBlur(img, (5, 5), 0)  # 5*5高斯核去噪,再otsu二值化
ret3, th3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

images = [img, 0, th1, img, 0, th2, blur, 0, th3]
titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v = 127)',
          'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
          'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
for i in range(3):
    plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
    plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)  # hist绘制直方图, ravel转置成一维矩阵
    plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
    plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
plt.show()
