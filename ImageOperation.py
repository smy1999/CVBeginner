import cv2

img = cv2.imread('image.jpg')
(b, g, r) = cv2.split(img)
img_merge = cv2.merge((r, g, b))
img = cv2.imread('image.jpg')

res1 = img + img_merge  # 取模操作 250 + 10 = 260 % 256 = 4
res2 = cv2.add(img, img_merge)  # 饱和操作 250 + 10 = 260 = 255

# 带权混合, 最后一个参数表示混合后每个图增加的灰度值
res3 = cv2.addWeighted(img, 0.3, img_merge, 0.7, 100)

"""
将logo覆盖到图片中
"""

opencv_logo = cv2.imread('opencv_logo.png')
desktop = cv2.imread('desktop.jpg')

# 提取desktop中与logo一样啊的部分
line, row, channel = opencv_logo.shape
roi = desktop[0:line, 0:row]

opencv_logo_gray = cv2.cvtColor(opencv_logo, cv2.COLOR_BGR2GRAY)
# 图像二值化, 图像/阈值/当图像大于阈值赋的值/方法, 返回ret阈值/mask矩阵
ret, masks = cv2.threshold(opencv_logo_gray, 180, 255, cv2.THRESH_BINARY)
masks_inv = cv2.bitwise_not(masks)

# mask表示仅在mask不为0的位置保留, mask为0的位置结果也为0
ans1 = cv2.bitwise_and(roi, roi, mask=masks)
ans2 = cv2.bitwise_and(opencv_logo, opencv_logo, mask=masks_inv)

dst = cv2.add(ans1, ans2)
desktop[0:line, 0:row] = dst
cv2.imshow('result', desktop)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""
cv2.THRESH_BINARY       大于阈值置255,小于置0
cv2.THRESH_BINARY_INV   大于阈值置0,小于置255
cv2.THRESH_TRUNC        大于阈值置阈值,小于不变
cv2.THRESH_TOZERO       小于阈值置0,大于不变
cv2.THRESH_TOZERO_INV   大于阈值置0,小于不变
"""