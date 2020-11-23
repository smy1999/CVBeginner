import cv2
import numpy as np


def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:  # 双击左键
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1)


img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('drawing')
cv2.setMouseCallback('drawing', draw_circle)  # 鼠标响应, 每次鼠标活动都会调用回调函数

while True:
    cv2.imshow('drawing', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
