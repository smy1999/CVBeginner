import cv2
import numpy as np

"""
颜色空间转换 cv2.cvtColor(img, flag)
常用flag : cv2.COLOR_BGR2GRAY / cv2.COLOR_BGR2HSV
"""

cap = cv2.VideoCapture('feedingCat.mp4')

while True:
    ret, frame = cap.read()
    if ret:
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 设定想要颜色的阈值
        lower_shrimp = np.array([0, 150, 150])
        upper_shrimp = np.array([30, 255, 255])

        # 仅保留图片中在上下两值区间内的部分
        mask = cv2.inRange(frame_hsv, lower_shrimp, upper_shrimp)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)

        k = cv2.waitKey(33) & 0xFF
        if k == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
