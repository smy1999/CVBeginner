import numpy as np
import cv2

"""
Draw OpenCV logo.
"""

cv2.namedWindow('test02')
img = np.zeros((512, 512, 3), np.uint8)

cv2.ellipse(img, (256, 160), (108, 108), 120, 0, 300, (0, 0, 255), -1)
cv2.ellipse(img, (128, 384), (108, 108), 0, 0, 300, (0, 255, 0), -1)
cv2.ellipse(img, (384, 384), (108, 108), -60, 0, 300, (255, 0, 0), -1)
cv2.circle(img, (256, 160), 40, (0, 0, 0), -1)
cv2.circle(img, (128, 384), 40, (0, 0, 0), -1)
cv2.circle(img, (384, 384), 40, (0, 0, 0), -1)

cv2.imshow('test02', img)
cv2.waitKey(0)
cv2.destroyAllWindows()