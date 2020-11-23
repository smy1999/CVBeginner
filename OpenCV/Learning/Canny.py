import cv2
from matplotlib import pyplot as plt
"""
边缘检测Canny算法原理:
1.边缘检测易受到噪声影响, 因此先使用5*5高斯滤波
2.对平滑后的图像使用Sobel算子计算水平和竖直方向的一阶导数(梯度)Gx和Gy,根据梯度找到边界梯度方向和大小
  梯度方向一般与边界垂直, 可能有: 垂直, 水平, 两个对角线方向
3.扫描政府图像, 检查每个点的梯度是不是周围具有相同梯度方向的点中最大的, 仅选取这部分点, 得到窄边界
4.设置两阈值minVal和maxVal:
  低于minVal的边界抛弃; 高于maxVal的边界保留; 介于两者之间看是否与真正的辩解相连决定是否保留
"""
img = cv2.imread('../src/image.jpg', 0)

# param: 图像/低阈值/高阈值/卷积核大小(默认3)/L2gradient(bool选择计算梯度的方程)
# True : grad = sqrt(Gx ^ 2 + Gy ^ 2)
# False: grad = abs(Gx ^ 2) + abs(Gy ^ 2)
ans = cv2.Canny(img, 100, 200)

plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(ans, cmap='gray'), plt.title('Canny Image')
plt.xticks([]), plt.yticks([])
plt.show()
