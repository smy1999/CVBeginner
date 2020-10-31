import cv2
import numpy as np

"""
根据选择的模式在拖动鼠标时绘制矩形或圆圈
"""

left_down = False
rectangle_or_circle = True
ix, iy = -1, -1


def draw_fun(event, x, y, flags, param):
    global ix, iy, left_down, rectangle_or_circle
    if event == cv2.EVENT_LBUTTONDOWN:
        left_down = True
        ix, iy = x, y  # 记录点击坐标
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        # 鼠标移动且左键按下情况
        if left_down:
            if rectangle_or_circle:
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                r = int(np.sqrt((x - ix) ** 2 + (y - iy) ** 2))
                cv2.circle(img, (x, y), r, (0, 0, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        left_down = False


img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('drawing')
cv2.setMouseCallback('drawing', draw_fun)
while True:
    cv2.imshow('drawing', img)
    k = cv2.waitKey(20) & 0xFF
    if k == ord('m'):
        rectangle_or_circle = not rectangle_or_circle
    elif k == 27:
        break
cv2.destroyAllWindows()
