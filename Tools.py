import cv2
import numpy as np


def cvt_rgb2hsv(r, g, b):
    """
    Convert from RGB to HSV
    :param r: R-level
    :param g: G-level
    :param b: B-level
    :return: hsv like [[[60, 255, 255]]]
    """
    arr = np.uint8([[[r, g, b]]])
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
    return hsv
