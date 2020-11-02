import numpy as np
import cv2

"""
cv2.warpAffine 接收尺寸为2*3的变换矩阵
cv2.wartPerspective 接收尺寸为3*3的变换矩阵
"""

img = cv2.imread('../src/image.jpg')

# 缩放
# 法1. 图像/输出图像尺寸(None)/x轴缩放系数/y轴缩放系数/缩放因子
res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# 法2. 图像/目标尺寸(宽, 高)/缩放因子
height, width = img.shape[:2]
res1 = cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)

# 平移
# [[1, 0, tx]
#  [0, 1, ty]] 表示沿x轴平移tx距离,沿y轴平移ty距离
Matrix = np.float32([[1, 0, 100], [0, 1, 50]])
# 图像/平移矩阵/新图像尺寸/borderValue=边界颜色
res2 = cv2.warpAffine(img, Matrix, (width + 100, height + 50), borderValue=(140, 150, 200))

# 翻转
res3 = cv2.flip(img, -1)  # 0垂直翻转/1水平翻转/-1水平垂直翻转

# 旋转
# 获取旋转矩阵, 旋转中心/旋转角度/缩放因子
Matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 45, 0.6)
res4 = cv2.warpAffine(img, Matrix, (2 * width, 2 * height))

# 仿射变换:原图中平行线结果图中也平行 需要三个点
position_original = np.float32([[50, 50], [200, 50], [50, 200]])  # 获取原图像和目标图像三个基准点的位置
position_objective = np.float32([[10, 100], [200, 50], [100, 250]])
Matrix = cv2.getAffineTransform(position_original, position_objective)  # 利用前后点位构建变换矩阵
res5 = cv2.warpAffine(img, Matrix, (width, height))

# 透视变换:原图中直线变换后还是直线 需要四个点
position_original = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])  # 获取四个基准点位置
position_objective = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
Matrix = cv2.getPerspectiveTransform(position_original, position_objective)  # 构建变换矩阵
res6 = cv2.warpPerspective(img, Matrix, (width, height))

cv2.imshow('img', img)
cv2.imshow('res1', res6)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
INTER_NEAREST   最近邻插值
INTER_LINEAR    双线性插值(默认)(扩展)
INTER_AREA      使用像素区域关系进行重采样(缩小)
INTER_CUBIC     4x4像素邻域的双三次插值(扩展但慢)
INTER_LANCZOS4  8x8像素邻域的Lanczos插值
"""