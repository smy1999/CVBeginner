import cv2
import numpy as np

"""
拼接苹果和橘子, 拼接处保证平滑
利用Gaussian金字塔将图像分辨率降低, 并构建Laplacian金字塔用于反构建图像
将分辨率最低的苹果橘子拼接
将各分辨率层级的苹果橘子的Laplacian金字塔拼接
利用最低层苹果橘子拼接图和各分辨率的Laplacian金字塔逐层构建出原分辨率的拼接图像
"""

# resize使其可以被2整除
apple = cv2.imread('../src/apple.png')
apple = cv2.resize(apple, (256, 256), interpolation=cv2.INTER_CUBIC)
orange = cv2.imread('../src/orange.png')
orange = cv2.resize(orange, (256, 256), interpolation=cv2.INTER_CUBIC)

# 生成2个Gaussian Pyramid
G = apple.copy()
apple_pyramid = [G]
for i in range(5):
    G = cv2.pyrDown(G)
    apple_pyramid.append(G)

G = orange.copy()
orange_pyramid = [G]
for i in range(5):
    G = cv2.pyrDown(G)
    orange_pyramid.append(G)

# 生成2个Laplacian Pyramid
apple_laplacian_pyramid = [apple_pyramid[5]]
for i in range(5, 0, -1):
    G = cv2.pyrUp(apple_pyramid[i])
    sub = cv2.subtract(apple_pyramid[i - 1], G)
    apple_laplacian_pyramid.append(sub)

orange_laplacian_pyramid = [orange_pyramid[5]]
for i in range(5, 0, -1):
    G = cv2.pyrUp(orange_pyramid[i])
    sub = cv2.subtract(orange_pyramid[i - 1], G)
    orange_laplacian_pyramid.append(sub)

# laplacian变换后的图像进行拼接
lap_after_merge = []
for orange_lap, apple_lap in zip(orange_laplacian_pyramid, apple_laplacian_pyramid):
    row, column, dpt = apple_lap.shape
    ans_lap = np.hstack((apple_lap[:, 0:column // 2], orange_lap[:, column // 2:]))
    lap_after_merge.append(ans_lap)

# 利用拼接的Laplacian反构建
ans_re = lap_after_merge[0]
for i in range(1, 6):
    ans_re = cv2.pyrUp(ans_re)
    ans_re = cv2.add(ans_re, lap_after_merge[i])

real = np.hstack((apple[:, 0:column // 2], orange[:, column // 2:]))


cv2.imshow('Pyramid', ans_re)
cv2.imshow("Real", real)
cv2.waitKey()
cv2.destroyAllWindows()
