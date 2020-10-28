import cv2
from matplotlib import pyplot as plt


img = cv2.imread('yue.png', 1)
ret, dst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

plt.subplot(121), plt.imshow(img), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('Threshold Binary Inv')
plt.xticks([]), plt.yticks([])
plt.show()

cv2.imwrite('yue_reverse.png', dst)
