import cv2

"""
加载灰度图, 显示图片, 按下's'保存后退出, 按下'esc'退出不保存
"""

img = cv2.imread('image.jpg', 0)
cv2.imshow('Read Lenna', img)
getKey = cv2.waitKey(0)
if getKey == ord('s'):  # ord(char) 返回char的Unicode值
    cv2.imwrite('Test01Saved.jpg', img)
    cv2.destroyAllWindows()
elif getKey == 27:  # esc
    cv2.destroyAllWindows()
