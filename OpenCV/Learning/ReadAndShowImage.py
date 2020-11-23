import cv2
"""
一种生成窗口的方法 :
先创建一个窗口, 再加载图像, 并可以决定窗口大小是否可调整
常用于图像维度大或添加轨迹条时
"""

# 创建窗口, WINDOW_NORMAL表示窗口大小可调, WINDOW_AUTOSIZE表示窗口大小不可调
cv2.namedWindow('window 1', cv2.WINDOW_NORMAL)
cv2.namedWindow('window 2', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('window 3', cv2.WINDOW_AUTOSIZE)

# 读取图像
img_default = cv2.imread("../src/image.jpg")  # 彩色
img0 = cv2.imread("../src/image.jpg", 0)  # 灰度
img1 = cv2.imread("../src/image.jpg", 1)  # 彩色

# 显示图像
# 若要在此前窗口打开, 此处窗口应同名
cv2.imshow('window 1', img0)
cv2.imshow('window 2', img_default)
cv2.imshow("window 3", img1)

# delay:毫秒 函数等待delay毫秒, 返回这时间内按键的ASCII码, 若没有按下则返回-1
# 若delay = 0, 则无限等待键盘输入, 可以用于检测按键是否被按下
cv2.waitKey()
cv2.destroyAllWindows()  # 关闭生成窗口

# 存储图片
cv2.imwrite('../src/saveLenna.png', img0)
