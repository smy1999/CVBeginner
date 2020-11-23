import cv2
import numpy as np

"""
直方图反向投影:用于图像分割或在图像中找感兴趣的部分
输出与输入图像同样大小的图像,其中每个像素代表了与输入图像对应点属于同样目标对象的概率
输出图像中像素值越高(白)的点越是搜索目标
常与Camshift算法一起使用

首先需要包含待查找目标的直方图,待查找对象尽量占满整张图像,将该图像投影到输入图像寻找目标,得到概率图像,设置恰当阈值进行二值化
"""

# 1.Numpy
roi = cv2.imread('../../src/rose_red.jpg')
target = cv2.imread('../../src/rose.jpg')
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
target_hist = cv2.calcHist([target_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

# 计算比值R = M / I, 反向投影R, 根据R创建新图像, 其中每个像素代表该点是目标的概率
np.seterr(divide='ignore', invalid='ignore')  # 忽略下面出现除0的错误
R = roi_hist / target_hist
h, s, v = cv2.split(target_hsv)
B = R[h.ravel(), s.ravel()]  # 得到像素一一对应的概率矩阵
B = np.minimum(B, 1)
B = B.reshape(target_hsv.shape[:2])

# 用原型卷积核卷积
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 返回指定结构和尺寸的结构元素, 形状/尺寸/锚点位置
B = cv2.filter2D(B, -1, disc)
B = np.uint8(B)
cv2.normalize(B, B, 0, 255, cv2.NORM_MINMAX)  # 归一化, 输入数组/输出数组/下限/上限/方法

ret, thresh = cv2.threshold(B, 50, 255, 0)
cv2.imshow('numpy', thresh)


# 2.cv2
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)  # 归一化方便显示
# 目标图像/通道/利用的直方图/范围/scale
dst = cv2.calcBackProject([target_hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
dst = cv2.filter2D(dst, -1, disc)  # 卷积将分散的点连在一起
ret1, thresh = cv2.threshold(dst, 50, 255, 0)
thresh = cv2.merge((thresh, thresh, thresh))  # 三通道
res = cv2.bitwise_and(target, thresh)  # 二值化图像与原图进行掩模处理
res = np.hstack((target, thresh, res))
cv2.imshow('cv2', res)

cv2.waitKey(0)
cv2.destroyAllWindows()
