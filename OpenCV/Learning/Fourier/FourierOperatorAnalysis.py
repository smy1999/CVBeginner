import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1.平均滤波器simple averaging filter
mean_filter = np.ones((3, 3))

# 2.高斯滤波器Gaussian filter
x = cv2.getGaussianKernel(5, 10)  # 计算高斯核, size/sigma
gaussian_filter = x * x.T

# 3.scharr x方向
scharr = np.array([[-3, 0, 3],
                   [-10, 0, 10],
                   [-3, 0, 3]])

# 4.sobel x方向
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

# 5.sobel y方向
sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# 6.拉普拉斯laplacian
laplacian = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])

filters = [mean_filter, gaussian_filter, scharr, sobel_x, sobel_y, laplacian]
filter_names = ['Mean Filter', 'Gaussian Filter', 'Scharr', 'Sobel X', 'Sobel Y', 'Laplacian']
fft_filters = [np.fft.fft2(x) for x in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
magnitude_spectrum = [np.log(np.abs(z) + 1) for z in fft_shift]


# 白色通过, 中心是低频部分
for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(magnitude_spectrum[i], 'gray')
    plt.title(filter_names[i]), plt.xticks([]), plt.yticks([])
plt.show()
