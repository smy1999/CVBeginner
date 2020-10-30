import cv2

img = cv2.imread('graffiti.png')
img_copy = img.copy()
img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 200, 255, 0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
dst = cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 3)

cv2.imshow('Threshold', thresh)
cv2.imshow('Contours', dst)

cnt = contours[1]
img_copy2 = img.copy()
contours_1 = cv2.drawContours(img_copy2, cnt, 1, (0, 255, 0), 3)
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




cv2.waitKey(0)
cv2.destroyAllWindows()
