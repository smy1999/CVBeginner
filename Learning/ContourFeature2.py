import cv2
import numpy as np

img = cv2.imread('../src/graffiti.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 200, 255, 0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[1]
img_cnt = cv2.drawContours(img.copy(), contours, 1, (0, 255, 0), 3)
cv2.imshow('Contours 1', img_cnt)

# 12.掩模
mask = np.zeros(img_gray.shape, np.uint8)
cv2.drawContours(mask, [cnt], 0, 255, -1)  # 填充
# np.nonzero 得到不为0的点的坐标,返回的是n行m列的(n为原数组维数, m为不为零的点的个数), 对应看
# np.transpose 转换为m行n列, 每行都是一个坐标
pixel_points = np.transpose(np.nonzero(mask))  # 得到掩模像素点
pixel_points2 = cv2.findNonZero(mask)  # 法2
# 最大值最小值和他们的位置
min_val, max_val, min_location, max_location = cv2.minMaxLoc(img_gray, mask=mask)
# 极点
left_most = tuple(cnt[cnt[:, :, 0].argmin()][0])
right_most = tuple(cnt[cnt[:, :, 0].argmax()][0])
top_most = tuple(cnt[cnt[:, :, 1].argmin()][0])
bottom_most = tuple(cnt[cnt[:, :, 1].argmax()][0])
img_copy1 = img.copy()
cv2.circle(img_copy1, left_most, 3, (0, 0, 255), -1)
cv2.circle(img_copy1, right_most, 3, (0, 0, 255), -1)
cv2.circle(img_copy1, top_most, 3, (0, 0, 255), -1)
cv2.circle(img_copy1, bottom_most, 3, (0, 0, 255), -1)
cv2.imshow('Draw Most Points', img_copy1)
cv2.imshow('Mask', mask)

# 13.求解图像中一个点到一个对象轮廓的最短距离
# param : 轮廓/坐标点/计算最短距离或判断位置关系(仅1,-1,0), 若在外部返回负,在轮廓上返回0,在内部为正
dist = cv2.pointPolygonTest(cnt, (50, 50), True)

# 14.形状匹配 根据两轮廓的Hu矩计算, 结果越小越好
# 两轮廓/矩的计算方法/默认为0
shape_match = cv2.matchShapes(cnt, cnt, cv2.CONTOURS_MATCH_I1, 0.0)


cv2.waitKey(0)
cv2.destroyAllWindows()
