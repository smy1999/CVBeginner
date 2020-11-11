import cv2

cap = cv2.VideoCapture("feedingCat.mp4")  # 文件读取

while True:
    ret, frame = cap.read()
    if ret:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame_gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:  # 视频读完后ret = False
        break
cap.release()
cv2.destroyAllWindows()
