import cv2
import numpy as np

cv2.namedWindow('drawing')
# uint8为图像数组格式
# 创建三维数组,长/宽/灰度
img = np.zeros((512, 512, 3), np.uint8)
# 图片/起始像素/终止像素/颜色RGB/宽度
cv2.line(img, (256, 0), (511, 511), (255, 0, 0), 5)

cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 2)

cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)  # 圆心/半径/-1表示填充

cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 60, (0, 255, 0), -1)  # 圆心/长轴短轴长/旋转角度/起始角度/终止角度/颜色RGB/宽度

pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)  # 须为int32型
pts = pts.reshape((-1, 1, 2))  # 须构建一个大小 : 行(顶点数)*1*2 的数组, -1表示模糊, 自动计算得出
img = cv2.polylines(img, [pts], True, (0, 255, 255), 3)  # 填充多条线, 顶点列表/是否闭合/颜色RGB/宽度

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'SMY', (100, 500), font, 5, (255, 255, 255), 2)  # 图片/内容/位置/字体/大小/颜色/宽度

cv2.imshow('drawing', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
