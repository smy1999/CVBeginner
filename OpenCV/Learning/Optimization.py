import cv2

e1 = cv2.getTickCount()  # 获取时钟数
summary = 0
for i in range(1, 100):
    summary += i
e2 = cv2.getTickCount()  # 获取时钟数
time = (e2 - e1) / cv2.getTickFrequency()  # 时钟数/每秒时钟频率得程序运行时间
print("程序执行时间 : " + str(time))

if cv2.useOptimized():  # 优化是否开启
    print("Optimization on.")
else:
    print("Optimization off.")
cv2.setUseOptimized(False)  # 关闭优化
cv2.setUseOptimized(True)  # 开启优化
