import cv2
import numpy as np

"""
轮廓Tips:
1.使用二值化图像. 寻找轮廓之前, 进行阈值化处理或Canny边界检测
2.查找轮廓会修改原图像
3.轮廓是指黑色背景中的白色物体的轮廓
"""
img = cv2.imread('../src/graffiti.png')
img_copy = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 200, 255, 0)

# 获取轮廓
# param : 图像/轮廓检索模式/轮廓近似方法
# return: 轮廓(是Python列表, 每个轮廓都是Numpy数组, 包含对象边界点的坐标)/轮廓层析结构
# cv2.CHAIN_APPROX_SIMPLE   若边界位置线存储边界的端点, 除冗余点, 节约内存
# cv2.CHAIN_APPROX_NONE     存储边界的每一个点
contours_simple, hierarchy_simple = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_none, hierarchy_none = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# 绘制轮廓
# param: 图像/轮廓/轮廓层次索引(-1表示所有轮廓)/轮廓颜色/轮廓厚度(-1表示填充)
dst_simple = cv2.drawContours(img, contours_simple, -1, (0, 255, 0), 3)
dst_none = cv2.drawContours(img_copy, contours_none, -1, (0, 255, 0), 3)
print('simple length : ' + str(cv2.arcLength(contours_simple[2], True)))
print('none length : ' + str(cv2.arcLength(contours_none[2], True)))

cv2.imshow('threshold', thresh)
cv2.imshow('Contour None', dst_none)
cv2.imshow('Contour Simple', dst_simple)
cv2.waitKey(0)
cv2.destroyAllWindows()
