import cv2
import numpy as np

img = cv2.imread('../src/graffiti.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 200, 255, 0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_copy = img.copy()
dst = cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 3)

cv2.imshow('Threshold', thresh)
cv2.imshow('Contours', dst)

cnt = contours[1]
img_copy2 = img.copy()
contours_1 = cv2.drawContours(img_copy2, contours, 1, (0, 255, 0), 3)
cv2.imshow('Contours 1', contours_1)

# 1.矩Moment
Moment = cv2.moments(cnt)
print(Moment)
# 利用矩计算对象重心
cx = int(Moment['m10'] / Moment['m00'])
cy = int(Moment['m01'] / Moment['m00'])
print("Barycenter : (" + str(cx) + " , " + str(cy) + ")")

# 2.面积Square
square = cv2.contourArea(cnt)
square2 = Moment['m00']  # 两种方法
print("Square : " + str(square) + " or " + str(square2))
equivalent_diameter = np.sqrt(4 * square / np.pi)  # 与矩阵面积想等的圆的直径
print("与矩阵面积想等的圆的直径 %f" % equivalent_diameter)

# 3.周长Perimeter 第二个参数用于指定对象形状是否闭合
perimeter = cv2.arcLength(cnt, True)
print("Perimeter : " + str(perimeter))

# 4.轮廓近似 Douglas-Peucker算法
epsilon = 0.001 * cv2.arcLength(cnt, True)  # 精确度, 越小越精确
approx = cv2.approxPolyDP(cnt, epsilon, True)  # 得到拟合的点集, 轮廓/精确度/是否闭合
img_copy3 = img.copy()
cv2.polylines(img_copy3, [approx], True, (255, 0, 0), 3)  # 根据点集连线
img_dp = cv2.drawContours(img_copy3, approx, -1, (0, 0, 255), 5)  # 画点集
cv2.imshow('Douglas-Peucker', img_dp)

# 5.凸包Hull 寻找最小凸集
# param:轮廓(/输出/是否顺时针/返回凸包上点的坐标或凸包对应轮廓上的点)
hull = cv2.convexHull(cnt)  # 获得凸包点集
img_copy4 = img.copy()
cv2.drawContours(img_copy4, contours, 1, (0, 255, 0), 3)
cv2.polylines(img_copy4, [hull], True, (255, 0, 0), 3)
img_hull = cv2.drawContours(img_copy4, hull, -1, (0, 0, 255), 5)  # 画凸包点集
cv2.imshow('hull', img_hull)
solidity = float(square) / cv2.contourArea(hull)  # 轮廓面积与凸包面积之比
print("轮廓面积与凸包面积之比 %f" % solidity)

# 6. 凸性缺陷 的点
hull_defect = cv2.convexHull(cnt, returnPoints=False)
defects = cv2.convexityDefects(cnt, hull_defect)  # 凸性缺陷点集(起点/终点/最远点/到最远点的距离, 前三个均为索引)
for i in range(defects.shape[0]):  # 每行
    p = defects[i, 0, 2]  # 取最远点的索引, 用cnt[p]取坐标
    cv2.drawContours(img_copy4, [cnt[p]], -1,  (0, 0, 255), 5)  # 绘制坐标
cv2.imshow('Hull defect', img_copy4)

# 7. 凸性检测 检测曲线是否是凸的
if cv2.isContourConvex(cnt):
    print("是凸曲线")
else:
    print("不是凸曲线")

# 8.边界矩形
# 直边界矩形 : return : 左上角坐标/宽/高
x, y, w, h = cv2.boundingRect(cnt)  # 得到外接矩形的坐标和尺寸
aspect_ratio = float(w) / h  # 外接矩形宽高比
print("边界矩形宽高比 %f" % aspect_ratio)
extent = float(square) / (w * h)  # 轮廓面积与矩形面积比
print("轮廓面积与矩形面积之比 %f" % extent)
img_copy5 = cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)
# 旋转边界矩形 : return : 左上角坐标/宽/高/旋转角度
rect = cv2.minAreaRect(cnt)  # 得到旋转矩形信息
box = cv2.boxPoints(rect)  # 返回Box2D型变量, 包含角点信息
box = np.int0(box)
cv2.drawContours(img_copy5, [box], -1, (255, 0, 0), 2)
cv2.imshow('rectangle', img_copy5)

# 9.最小外接圆
(x, y), r = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
r = int(r)
img_copy6 = cv2.circle(img.copy(), center, r, (0, 255, 0), 2)
cv2.imshow('Enclosing Circle', img_copy6)

# 10.椭圆拟合(旋转边界矩形的内切圆)
ellipse = cv2.fitEllipse(cnt)
img_copy7 = cv2.ellipse(img.copy(), ellipse, (0, 255, 0), 2)
cv2.imshow('Fitting Ellipse', img_copy7)
print("对象的方向(椭圆的方向) %f, %f" % ellipse[0])
print("椭圆短轴长%f, 长轴长%f" % (ellipse[1]))

# 11.直线拟合
row, column = img.shape[:2]
# param : 待拟合的直线点集/距离类型/距离值(与类型有关)/径向和角度精度(默认0.01)
# return: 曲线方向/曲线上一点, 点斜式表示直线
[vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
left_y = int((-x * vy / vx) + y)  # 获取图像最左的y
right_y = int(((column - x) * vy / vx) + y)  # 获取图像最右的y
img_copy8 = cv2.line(img, (column - 1, right_y), (0, left_y), (0, 255, 0), 2)
cv2.imshow('Fitting Line', img_copy8)
"""
cv2.DIST_USER       User defined distance
cv2.DIST_L1         abs(x1 - x2) + abs(y1 - y2)
cv2.DIST_L2         欧式距离, 此时与最小二乘法相同
cv2.DIST_C          max(abs(x1 - x2), abs(y1 - y2))
cv2.DIST_L12        2 * (sqrt(1 + x ^ 2 / 2) - 1))
cv2.DIST_FAIR       c ^ 2 * (abs(x) / c- log(1 + abs(x) / c)), c = 1.3998
cv2.DIST_WELSCH     c2 / 2 * (1 - exp(-(x / c)2)), c = 2.9846
cv2.DIST_HUBER      abs(x) < c ? x ^ 2 / 2 : c * (abs(x) - c / 2), c=1.345
"""

cv2.waitKey(0)
cv2.destroyAllWindows()
