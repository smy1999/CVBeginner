import cv2
import numpy as np

"""
创建画板, 使用各种自选颜色画笔绘制各种图形
"""


def nothing(x):
    pass


left_down = False
rectangle_or_circle = True
ix, iy = -1, -1


def draw_fun(event, x, y, flags, param):
    global ix, iy, left_down, rectangle_or_circle

    r = cv2.getTrackbarPos('R', 'Drawing Box')
    g = cv2.getTrackbarPos('G', 'Drawing Box')
    b = cv2.getTrackbarPos('B', 'Drawing Box')
    color = (b, g, r)
    if event == cv2.EVENT_LBUTTONDOWN:
        left_down = True
        ix, iy = x, y  # 记录点击坐标
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        # 鼠标移动且左键按下情况
        if left_down:
            if rectangle_or_circle:
                cv2.rectangle(img, (ix, iy), (x, y), color, -1)
            else:
                r = int(np.sqrt((x - ix) ** 2 + (y - iy) ** 2))
                cv2.circle(img, (x, y), r, color, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        left_down = False


img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('Drawing Box')

cv2.createTrackbar('R', 'Drawing Box', 0, 255, nothing)
cv2.createTrackbar('G', 'Drawing Box', 0, 255, nothing)
cv2.createTrackbar('B', 'Drawing Box', 0, 255, nothing)

# 每次对鼠标进行操作都会调用回调函数
cv2.setMouseCallback('Drawing Box', draw_fun)

while True:
    cv2.imshow('Drawing Box', img)
    k = cv2.waitKey(20) & 0xFF
    if k == ord('m'):
        rectangle_or_circle = not rectangle_or_circle
    elif k == 27:
        break
cv2.destroyAllWindows()
