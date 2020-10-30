import cv2

img = cv2.imread('graffiti.png')
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
cv2.polylines(img_copy4, [hull], True, (255, 0, 0), 3)
img_hull = cv2.drawContours(img_copy4, hull, -1, (0, 0, 255), 5)  # 画凸包点集
print(hull)
cv2.imshow('hull', img_hull)

# 6. 凸性缺陷
hull_defect = cv2.convexHull(cnt, returnPoints=False)
defects = cv2.convexityDefects(cnt, hull_defect)  # 凸性缺陷点集
print(defects)
cv2.drawContours(img_copy4, [defects.reshape(-1, 1, 2)], -1, (0, 0, 255), 5)  # 画凸性缺陷点集
cv2.imshow('Hull defect', img_copy4)


cv2.waitKey(0)
cv2.destroyAllWindows()
