import cv2
import numpy as np
from matplotlib import pyplot as plt

pos = np.array([[10, 50],
                [12, 49],
                [11, 52],
                [13, 52.2],
                [12.9, 50]], np.float32)  # 测量值矩阵

kalman = cv2.KalmanFilter(2, 2)  # 状态空间维数/测量值维数/控制变量维数

kalman.measurementMatrix = np.array([[1, 0], [0, 1]], np.float32)  # 测量矩阵
kalman.transitionMatrix = np.array([[1, 0], [0, 1]], np.float32)  # 状态转移矩阵
kalman.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1e-3  # 模型系统协方差矩阵
kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.01  # 测量系统协方差矩阵

kalman.statePre = np.array([[6], [6]], np.float32)

for i in range(len(pos)):
    mes = np.reshape(pos[i, :], (2, 1))  # 将pos每一行转换成两行一列
    x = kalman.correct(mes)
    y = kalman.predict()
    print(kalman.statePre[0], kalman.statePre[1])  # 预测状态
    print(kalman.statePost[0], kalman.statePost[1])  # 矫正状态
    print('measurement :\t', mes[0], mes[1])
    print('correct :\t', x[0], x[1])
    print('predict :\t', y[0], y[1])
    print('=' * 30)
