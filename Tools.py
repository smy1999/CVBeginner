import cv2
import numpy as np


def cvt_rgb2hsv(r, g, b):
    """
    Convert from RGB to HSV.
    :param r: R-level
    :param g: G-level
    :param b: B-level
    :return: hsv like [[[60, 255, 255]]]
    """
    arr = np.uint8([[[r, g, b]]])
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
    return hsv


def color_inverse(filename, flag):
    """
    Inverse the given image file.
    :param filename: Target filename
    :param flag: If true, save the result file
    :return: Result file
    """
    img = cv2.imread(filename, 1)
    ret, dst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    if flag:
        cv2.imwrite('color_inverse_result.png', dst)
    return dst
