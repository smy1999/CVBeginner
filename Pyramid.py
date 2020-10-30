from matplotlib import pyplot as plt
import cv2
"""
用于对同一图像的不同分辨率的子图像进行处理.
例如在图像中查找某个目标, 但不知道目标在图像中的尺寸大小.
需要创建一组图像, 是具有不同分辨率的原始图像, 将小分辨率置于顶部, 大分辨率置于底部, 为图像金字塔
"""
img = cv2.imread('image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Gaussian Pyramid: 顶部是由底部图像中连续的行和列去除得到的, 顶部图像中每个像素等于下一层五个像素的高斯加权平均
lower_reso1 = cv2.pyrDown(img)  # 尺寸变小, 分辨率降低
higher_reso1 = cv2.pyrUp(lower_reso1)  # 尺寸变大, 分辨率不变
lower_reso2 = cv2.pyrDown(higher_reso1)
higher_reso2 = cv2.pyrUp(lower_reso2)
lower_reso3 = cv2.pyrDown(higher_reso2)
lower_reso4 = cv2.pyrDown(lower_reso3)
lower_reso5 = cv2.pyrDown(lower_reso4)
lower_reso6 = cv2.pyrDown(lower_reso5)

plt.subplot(331), plt.imshow(img), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(332), plt.imshow(lower_reso1), plt.title('Lower1 Image')
plt.xticks([]), plt.yticks([])
plt.subplot(333), plt.imshow(higher_reso1), plt.title('Expand Lower1 Image')
plt.xticks([]), plt.yticks([])
plt.subplot(334), plt.imshow(lower_reso2), plt.title('Lower2 Image')
plt.xticks([]), plt.yticks([])
plt.subplot(335), plt.imshow(higher_reso2), plt.title('Expand Lower2 Image')
plt.xticks([]), plt.yticks([])
plt.subplot(336), plt.imshow(lower_reso3), plt.title('Lower3 Image')
plt.xticks([]), plt.yticks([])
plt.subplot(337), plt.imshow(lower_reso4), plt.title('Lower4 Image')
plt.xticks([]), plt.yticks([])
plt.subplot(338), plt.imshow(lower_reso5), plt.title('Lower5 Image')
plt.xticks([]), plt.yticks([])
plt.subplot(339), plt.imshow(lower_reso6), plt.title('Lower6 Image')
plt.xticks([]), plt.yticks([])
plt.show()
