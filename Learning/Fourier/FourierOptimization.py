import numpy as np
import cv2

"""
数组大小为2的指数时DFT效率最高, 其次是2/3/5的倍数时效率也高
为了提升效率, 修改图像大小(补0): Numpy自动补0, cv2手动补0
"""

img = cv2.imread('../../src/rose.jpg', 0)
row, column = img.shape
print('row = %d, column = %d' % (row, column))
row_better = cv2.getOptimalDFTSize(row)
column_better = cv2.getOptimalDFTSize(column)
print('Optimal row = %d, column = %d' % (row_better, column_better))

# 构建新尺寸的原图
# 法1.
img_better = np.zeros((row_better, column_better))
img_better[:row, :column] = img
# 法2.
right_border = column_better - column
bottom_border = row_better - row
img_better2 = cv2.copyMakeBorder(img, 0, bottom_border, 0, right_border, cv2.BORDER_CONSTANT, value=0)