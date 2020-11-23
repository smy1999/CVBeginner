import cv2
from matplotlib import pyplot as plt

"""
使用plt显示图像
"""

# opencv加载BGR模式, matplotlib加载RGB模式, 若opencv读取彩色图像, 则matplotlib读取会有反色
img = cv2.imread('../src/image.jpg', 0)
# cmap : 将标量数据映射到色彩图
# interpolation : 插值方法
# bicubic 双三次插值
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([])  # x轴标尺, 此时无标尺
plt.yticks([])  # y轴标尺, 此时无标尺
plt.show()  # 显示图像
