import numpy as np
import cv2
from matplotlib import pyplot as plt

"""
梯度实际就是求导
高通滤波器:Sobel, Scharr, Laplacian
Sobel, Scharr即求一阶或二阶导数, Scharr是对Sobel(使用小的卷积和求解梯度角度时的优化)
Laplacian是求二阶导数
"""
img = cv2.imread('../src/yue_reverse.png', 0)
"""
Sobel算子是高斯平滑与微分的结合, 抗噪能力强, 可设定求导方向和卷积核的大小(-1则使用3*3的Scharr,效果好于3*3Sobel)
Laplacian算子使用二阶导数定义, 假设离散实现类似二阶Sobel导数
3*3 Scharr x方向   3*3 Scharr y方向         Laplacian
[-3, 0, 3]        [-3, -10, -3]            [0, 1, 0]
[-10, 0, 10]      [0, 0, 0]                [1, -4, 1]
[-3, 0, 3]        [3, 10, 3]               [0, 1, 0]
"""
# cv2.CV_64F输出图像深度, -1表示与原图一致(np.uint8)
laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # 1,0表示只在x方向求一阶导, 最大2阶
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # 0,1表示只在y方向求一阶导, 最大2阶

# 若直接取8u会造成白黑边界负数, 进而截断为0, 丢失单边数据
# 先用高精度做,再取绝对值转换回低精度
sobelx64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
abs_sobelx64f = np.absolute(sobelx64f)
sobelx8u = np.uint8(abs_sobelx64f)
sobely64f = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
abs_sobely64f = np.absolute(sobely64f)
sobely8u = np.uint8(abs_sobely64f)

sobelx8u_direct = cv2.Sobel(img, -1, 1, 0, ksize=5)
sobely8u_direct = cv2.Sobel(img, -1, 0, 1, ksize=5)

plt.subplot(331), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(332), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian')
plt.xticks([]), plt.yticks([])
plt.subplot(334), plt.imshow(sobel_x, cmap='gray'), plt.title('Sobel x')
plt.xticks([]), plt.yticks([])
plt.subplot(337), plt.imshow(sobel_y, cmap='gray'), plt.title('Sobel y')
plt.xticks([]), plt.yticks([])
plt.subplot(335), plt.imshow(sobelx8u, cmap='gray'), plt.title('Sobel x 8uint')
plt.xticks([]), plt.yticks([])
plt.subplot(336), plt.imshow(sobelx8u_direct, cmap='gray'), plt.title('Sobel x 8uint direct')
plt.xticks([]), plt.yticks([])
plt.subplot(338), plt.imshow(sobely8u, cmap='gray'), plt.title('Sobel y 8uint')
plt.xticks([]), plt.yticks([])
plt.subplot(339), plt.imshow(sobely8u_direct, cmap='gray'), plt.title('Sobel y 8uint direct')
plt.xticks([]), plt.yticks([])
plt.show()
