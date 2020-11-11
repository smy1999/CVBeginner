import cv2
import numpy as np

"""
卡尔曼滤波预测鼠标移动
R预测 G测量
"""

frame = np.zeros((800, 800, 3), np.uint8)
last_mes = current_mes = np.array((2, 1), np.float32)
last_pre = current_pre = np.array((2, 1), np.float32)


def mouse_move(event, x, y, s, p):
    global frame, current_mes, current_pre, last_mes, last_pre
    last_pre = current_pre
    last_mes = current_mes
    current_mes = np.array([[np.float32(x)], [np.float32(y)]])

    kalman.correct(current_mes)
    current_pre = kalman.predict()

    last_mes_x, last_mes_y = int(last_mes[0]), int(last_mes[1])
    last_pre_x, last_pre_y = int(last_pre[0]), int(last_pre[1])
    current_mes_x, current_mes_y = int(current_mes[0]), int(current_mes[1])
    current_pre_x, current_pre_y = int(current_pre[0]), int(current_pre[1])
    cv2.line(frame, (last_mes_x, last_mes_y), (current_mes_x, current_mes_y), (0, 255, 0))
    cv2.line(frame, (last_pre_x, last_pre_y), (current_pre_x, current_pre_y), (0, 0, 255))


cv2.namedWindow("Kalman Filter")
cv2.setMouseCallback("Kalman Filter", mouse_move)
kalman = cv2.KalmanFilter(4, 2)  # 四维，横纵坐标/横纵速度
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)  # 测量值横纵坐标
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.003
kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32)

while True:
    cv2.imshow('Kalman Filter', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
