from matplotlib import pyplot as plt
import numpy as np
import cv2

"""
2D离散傅立叶变换DFT分析图像的频域特性, 实现的方法之一快速傅立叶变换FFT
边界和噪声变化快,属于高频分量
"""
img = cv2.imread('../../src/desktop.jpg', 0)

# 1.Numpy
# 快速傅立叶变换, param:图像(gray)/输出数组大小(缺省与输入相同;若结果比输入大,输入图像在FFT前补0;反之输入图像被切割)
fourier = np.fft.fft2(img)
# 此时频率为0(直流分量)在输出图像左上角, 故平移
fourier_shift = np.fft.fftshift(fourier)

magnitude_spectrum = 20 * np.log(np.abs(fourier_shift))  # 构建振幅谱

row, column = img.shape
row_half, column_half = row // 2, column // 2
fourier_shift[row_half - 30: row_half + 30, column_half - 30: column_half + 30] = 0  # 掩模,去除低频分量
fourier_ishift = np.fft.ifftshift(fourier_shift)  # 中心低频分量返回左上角
img_back = np.fft.ifft2(fourier_ishift)  # 反傅立叶变换
img_back = np.abs(img_back)  # 取绝对值

plt.subplot(221), plt.imshow(img, 'gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(magnitude_spectrum, 'gray'), plt.title('Magnitude Spectrum')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(img_back, 'gray'), plt.title('Image after HPF')  # 高通滤波
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img_back), plt.title('Image in JET')
plt.xticks([]), plt.yticks([])
plt.show()

# 2.cv2(更快)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)  # 输入float32格式图像
dft_shift = np.fft.fftshift(dft)
# magnitude计算二维矢量的幅值, param: 实部/虚部
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

mask = np.zeros((row, column, 2), np.uint8)
mask[row_half - 30: row_half + 30, column_half - 30: column_half + 30] = 1  # 掩模
fourier_shift2 = mask * dft_shift
fourier_ishift2 = np.fft.ifftshift(fourier_shift2)  # 反平移回左上角
img_back2 = cv2.idft(fourier_ishift2)  # 反变换
img_back2 = cv2.magnitude(img_back2[:, :, 0], img_back2[:, :, 1])

plt.subplot(221), plt.imshow(img, 'gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(magnitude_spectrum, 'gray'), plt.title('Magnitude Spectrum')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(img_back2, 'gray'), plt.title('Image after HPF')  # 高通滤波
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img_back2), plt.title('Image in JET')
plt.xticks([]), plt.yticks([])
plt.show()
