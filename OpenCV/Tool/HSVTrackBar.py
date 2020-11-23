import cv2
import numpy as np


def nothing(x):
    pass


img = np.zeros((256, 256, 3), np.uint8)
cv2.namedWindow('HSVTrackBar')

cv2.createTrackbar('H', 'HSVTrackBar', 0, 179, nothing)
cv2.createTrackbar('S', 'HSVTrackBar', 0, 255, nothing)
cv2.createTrackbar('V', 'HSVTrackBar', 0, 255, nothing)

while True:
    cv2.imshow('HSVTrackBar', img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    h = cv2.getTrackbarPos('H', 'HSVTrackBar')
    s = cv2.getTrackbarPos('S', 'HSVTrackBar')
    v = cv2.getTrackbarPos('V', 'HSVTrackBar')
    img[:] = [h, s, v]
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

cv2.destroyAllWindows()
