import cv2
import numpy as np
from matplotlib import pyplot as plt

# 低通滤波LPF去除噪音, 模糊图像
# 高通滤波HPF找到图像边缘

img = cv2.imread('coin_noise.jpg')

# 1.使用filter2D进行2D卷积
kernel = np.ones((5, 5), np.float32) / 25  # 构建5*5的平均滤波器核
dst = cv2.filter2D(img, -1, kernel)  # 卷积运算, 第二个param表示图像深度(-1表示和原图一致)

# 2.均值滤波 类似2D卷积
blur = cv2.blur(img, (5, 5))

# 3.方框滤波 图像/图像深度/方框大小/false表示不均值,加和后若大于255则置255
boxFilter = cv2.boxFilter(img, -1, (5, 5), normalize=False)

# 4.高斯模糊 卷积核为高斯核
# 需指定高斯核的宽和高(均为奇数), 沿x和y方向的标准差(指定一个另一个取相同值, 标准差为0则自适应计算)
# 可用cv2.getGaussianKernel()构建高斯核
gauss_blur = cv2.GaussianBlur(img, (5, 5), 0)  # 0表示根据窗口大小计算高斯函数标准差

# 5.中值模糊 卷积框的中位数作为像素中心的值
median = cv2.medianBlur(img, 5)

# 6.双边滤波, 同时使用空间高斯权重和灰度值相似性高斯权重,
# 前者确保只有临近区域的像素对中心点有影响, 后者确保只有与中心像素灰度值相近的才会用来进行运算
# 保证边界清晰情况下滤波 但较慢
# 图像/邻域直径/空间高斯函数标准差/灰度值相似性高斯函数标准差
bi_blur = cv2.bilateralFilter(img, 9, 75, 75)

plt.subplot(331), plt.imshow(img), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(332), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.subplot(333), plt.imshow(blur), plt.title('Blur')
plt.xticks([]), plt.yticks([])
plt.subplot(334), plt.imshow(boxFilter), plt.title('Box Filter')
plt.xticks([]), plt.yticks([])
plt.subplot(335), plt.imshow(gauss_blur), plt.title('Gaussian Blur')
plt.xticks([]), plt.yticks([])
plt.subplot(336), plt.imshow(median), plt.title('Median Blur')
plt.xticks([]), plt.yticks([])
plt.subplot(337), plt.imshow(bi_blur), plt.title('Bilateral Filter')
plt.xticks([]), plt.yticks([])
plt.show()
