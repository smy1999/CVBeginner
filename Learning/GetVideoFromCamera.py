import cv2

# 新建VideoCapture对象, 参数为设备索引号或视频文件, 0为默认摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():  # 检测摄像头是否初始化
    cap.open()  # 若初始化未成功, 打开摄像头
while True:
    # type(ret) = bool, 如果读取帧正确返回True,如果文件读取到结尾, 返回False
    # frame是三维矩阵, 表示每一帧的图像
    ret, frame = cap.read()
    # 颜色空间转换, BGR转换为灰度
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 显示摄像头
    cv2.imshow('frame', frame_gray)
    # 若1毫秒内有按下q则退出(先执行&, 防止bug)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 释放资源, 关闭摄像头或视频
cap.release()
cv2.destroyAllWindows()
