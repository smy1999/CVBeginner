import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../src/image.jpg')
img_gray = cv2.imread('../src/image.jpg', 0)

bgr = img[200, 200]  # 查看某点像素值
print("(200, 200) 的像素值为" + str(bgr))
b = img[200, 200, 0]  # 查看某点rgb单一像素值
print("(200, 200) 的b像素值为" + str(b))
img[200, 200, 2] = 255  # 修改某点像素值
img[200, 200] = [255, 255, 255]  # 修改某点像素值

# 另一种方法
b = img.item(200, 200, 0)  # 获取某点rgb单一像素值
print("(200, 200) 的b像素值为" + str(b))
img.itemset((200, 200, 0), 0)  # 修改

(row, column, channel) = img.shape  # 返回(行/列/通道数)的元组
print("img ：%d * %d * %d" % (row, column, channel))
(row, column) = img_gray.shape  # 灰度图像返回(行/列)
print("img_gray ：%d * %d" % (row, column))
s = img.size  # 返回像素数
print("img.size = " + str(s))
t = img.dtype  # 返回img的类型
print("img.dtype = " + str(t))

# 选取部分并拷贝
eye = img[254:280, 250:285]
img[50:76, 50:85] = eye

# 通道拆分与合并(费时)
b, g, r = cv2.split(img)  # 三通道拆分
img = cv2.merge((r, g, b))  # 三通道合并

# 通道拆分与合并(省时)
b = img[:, :, 2]
img[:, :, 2] = 0

# cv2.imshow('window', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 图像填充边界 常用语卷积或0填充
# 图像/上下左右高度/类型
img = cv2.imread('../src/image.jpg')
# 重复最后一个元素           aaaaaa|abcdefgh|hhhhhhh
replicate = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
# 边界元素镜像(有边元素)      fedcba|abcdefgh|hgfedcb
reflect = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_REFLECT)
# 边界元素镜像(无边元素)      gfedcb|abcdefgh|gfedcba
reflect101 = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_REFLECT_101)
# 重复相对方向              cdefgh|abcdefgh|abcdefg
wrap = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_WRAP)
# 固定颜色
constant = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 0, 0])

plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')

plt.show()

"""
def find_position(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("(%d, %d)" % (x, y))


cv2.namedWindow('window')
cv2.setMouseCallback('window', find_position)
while True:
    cv2.imshow('window', img)
    if cv2.waitKey(0) & 0xFF == 27:
        break
cv2.destroyAllWindows()
"""