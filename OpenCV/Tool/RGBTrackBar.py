import cv2
import numpy as np


def nothing(x):
    pass


img = np.zeros((256, 512, 3), np.uint8)
cv2.namedWindow('tb')

# 创建调色盘, 名称/窗口/范围/回调函数
cv2.createTrackbar('R', 'tb', 0, 255, nothing)
cv2.createTrackbar('G', 'tb', 0, 255, nothing)
cv2.createTrackbar('B', 'tb', 0, 255, nothing)

switch = '0:OFF\n1:ON'
cv2.createTrackbar(switch, 'tb', 0, 1, nothing)

while True:
    cv2.imshow('tb', img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    # 获取其数值
    r = cv2.getTrackbarPos('R', 'tb')
    g = cv2.getTrackbarPos('G', 'tb')
    b = cv2.getTrackbarPos('B', 'tb')
    s = cv2.getTrackbarPos(switch, 'tb')

    if s == 0:
        img[:] = 0
    else:
        # 颜色通道,img全体为bgr
        img[:] = [b, g, r]

cv2.destroyAllWindows()
