import cv2

"""
轮廓:黑中的白色物体的轮廓
contour轮廓由四元组表示(若没有则为-1):轮廓包含则有父子关系
Next: 同一级组织结构的下一个轮廓
Previous: 同一级组织结构的上一个轮廓
First_Child: 第一个字轮廓
Parent: 父轮廓
"""

img = cv2.imread('../../src/graffiti.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 200, 255, 0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
dst = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)

cv2.imshow('', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
cv2.RETR_LIST       所有轮廓平等,不创建父子关系,均属于同一级组织结构,用于不关心轮廓关系
cv2.RETR_TREE       返回完整的组织结构
cv2.RETR_CCOMP      返回两层组织结构,交替排序(对象外轮廓为1,内轮廓为2,内轮廓中的对象外轮廓为1)
cv2.RETR_EXTERNAL   只返回最外层的轮廓,忽略其他轮廓
"""