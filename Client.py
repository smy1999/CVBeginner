import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('src/desktop.jpg', 0)

# 2.cv2
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)  # 输入float32格式图像

dft_shift = np.fft.fftshift(dft)
# magnitude计算二维矢量的幅值, param: 实部/虚部

magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
row, column = img.shape
row_half, column_half = row // 2, column // 2

mask = np.zeros((row, column, 2), np.uint8)

mask[row_half - 30: row_half + 30, column_half - 30: column_half + 30] = 1  # 掩模
fourier_shift2 = dft_shift * mask
fourier_ishift2 = np.fft.ifftshift(fourier_shift2)  # 反平移回左上角

img_back2 = cv2.idft(fourier_ishift2)  # 反变换
img_back2 = cv2.magnitude(img_back2[:, :, 0], img_back2[:, :, 1])
img_back2 = cv2.idft(fourier_ishift2)
img_back2 = cv2.magnitude(img_back2[:,:,0], img_back2[:,:,1])
"""
plt.subplot(221), plt.imshow(img, 'gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(magnitude_spectrum, 'gray'), plt.title('Magnitude Spectrum')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(img_back2, 'gray'), plt.title('Image after HPF')  # 高通滤波
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img_back2), plt.title('Image in JET')
plt.xticks([]), plt.yticks([])
plt.show()
"""

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Input Image'),plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(img_back2,cmap = 'gray')
plt.title('Magnitude Spectrum'),plt.xticks([]),plt.yticks([])
plt.show()