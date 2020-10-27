import cv2

cap = cv2.VideoCapture(0)
# *使后面视为元组
# 确定编码格式
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 文件名/编码格式/帧数/尺寸
out = cv2.VideoWriter('saveVideo.avi', fourcc, 20.0, (640, 480))
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        out.write(frame)  # 写入
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
