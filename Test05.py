import cv2

"""
通过调节滑动条设置minVal和maxVal
"""

cv2.namedWindow('Client')
img = cv2.imread('image.jpg')
dst = img


def nothing(x):
    pass


cv2.createTrackbar('minVal', 'Client', 0, 255, nothing)
cv2.createTrackbar('maxVal', 'Client', 0, 255, nothing)

while True:
    cv2.imshow('Client', dst)
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break
    minVal = cv2.getTrackbarPos('minVal', 'Client')
    maxVal = cv2.getTrackbarPos('maxVal', 'Client')
    dst = cv2.Canny(img, minVal, maxVal)

cv2.destroyAllWindows()
